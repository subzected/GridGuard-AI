import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.database.models import MONITORING_LOCATIONS
from src.database.repository import upsert_weather_records


API_URL = "https://api.open-meteo.com/v1/forecast"

# Thunderstorm probability is requested for diagnostics, but the current
# Open-Meteo response has returned only NULL values for the monitored areas.
# Keep it optional and never replace missing values with zero.
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "pressure_msl",
    "precipitation_probability",
    "thunderstorm_probability",
    "weather_code",
]

MAX_RETRIES = 4
REQUEST_TIMEOUT_SECONDS = 30
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def _validate_hourly_response(hourly: Dict[str, Any]) -> int:
    """Validate required hourly fields and return the number of timestamps."""
    required = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "rain",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "pressure_msl",
        "precipitation_probability",
        "weather_code",
    ]

    missing = [field for field in required if field not in hourly]
    if missing:
        raise ValueError(
            f"Open-Meteo response is missing required fields: {missing}"
        )

    record_count = len(hourly["time"])
    if record_count == 0:
        raise ValueError("Open-Meteo returned an empty hourly forecast.")

    for field in required:
        if len(hourly[field]) != record_count:
            raise ValueError(
                f"Hourly field '{field}' has {len(hourly[field])} values; "
                f"expected {record_count}."
            )

    thunderstorm_values = hourly.get("thunderstorm_probability")
    if (
        thunderstorm_values is not None
        and len(thunderstorm_values) != record_count
    ):
        raise ValueError(
            "Hourly field 'thunderstorm_probability' does not match "
            "the number of timestamps."
        )

    return record_count


def fetch_weather(
    location: Dict[str, Any],
    fetched_at: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch one location's hourly forecast.

    fetched_at is supplied by collect_weather so all locations in the same
    collection cycle share exactly the same decision-time timestamp.
    """
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "forecast_days": 3,
        "timezone": "UTC",
    }

    response = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            break

        except requests.HTTPError as error:
            status = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if (
                status not in RETRYABLE_STATUS_CODES
                or attempt == MAX_RETRIES - 1
            ):
                raise

            wait_seconds = 2 ** (attempt + 1)
            logger.warning(
                "%s returned HTTP %s. Retrying in %s seconds "
                "(attempt %s/%s).",
                location["name"],
                status,
                wait_seconds,
                attempt + 2,
                MAX_RETRIES,
            )
            time.sleep(wait_seconds)

        except (requests.ConnectionError, requests.Timeout):
            if attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = 2 ** (attempt + 1)
            logger.warning(
                "Connection problem for %s. Retrying in %s seconds "
                "(attempt %s/%s).",
                location["name"],
                wait_seconds,
                attempt + 2,
                MAX_RETRIES,
            )
            time.sleep(wait_seconds)

    if response is None:
        raise requests.RequestException(
            f"Could not fetch weather for {location['name']}."
        )

    try:
        data = response.json()
    except ValueError as error:
        raise ValueError(
            f"Open-Meteo returned invalid JSON for {location['name']}."
        ) from error

    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(
            f"Hourly forecast missing for {location['name']}. "
            f"API response: {data}"
        )

    record_count = _validate_hourly_response(hourly)

    thunderstorm_values = hourly.get(
        "thunderstorm_probability",
        [None] * record_count,
    )

    frame = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(
                hourly["time"],
                utc=True,
            ).tz_localize(None),
            "temperature_c": hourly["temperature_2m"],
            "relative_humidity_pct": hourly["relative_humidity_2m"],
            "precipitation_mm": hourly["precipitation"],
            "rain_mm": hourly["rain"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
            "wind_gusts_kmh": hourly["wind_gusts_10m"],
            "wind_direction_deg": hourly["wind_direction_10m"],
            "pressure_hpa": hourly["pressure_msl"],
            "precipitation_probability_pct": hourly[
                "precipitation_probability"
            ],
            "thunderstorm_probability_pct": thunderstorm_values,
            "weather_code": hourly["weather_code"],
        }
    )

    frame["location_id"] = location["location_id"]
    frame["location_name"] = location["name"]
    frame["latitude"] = location["latitude"]
    frame["longitude"] = location["longitude"]

    # All locations in this polling cycle receive the same UTC timestamp.
    if fetched_at is None:
        fetched_at = datetime.now(timezone.utc).replace(tzinfo=None)

    frame["fetched_at"] = fetched_at
    frame["source"] = "open-meteo"
    frame["temporal_resolution"] = "hourly"

    available_thunderstorm_values = frame[
        "thunderstorm_probability_pct"
    ].notna().sum()

    logger.info(
        "%s: %s hourly records; thunderstorm probability available for %s",
        location["name"],
        len(frame),
        available_thunderstorm_values,
    )

    return frame


def collect_weather() -> int:
    """Fetch all monitoring locations and preserve one timestamp per cycle."""
    all_frames: List[pd.DataFrame] = []
    failed_locations: List[str] = []

    # Capture once, before fetching the first location.
    collection_time = datetime.now(timezone.utc).replace(tzinfo=None)

    logger.info(
        "Starting Open-Meteo collection for %s locations. Collection timestamp: %s UTC",
        len(MONITORING_LOCATIONS),
        collection_time.isoformat(sep=" "),
    )

    for location in MONITORING_LOCATIONS:
        try:
            logger.info("Fetching weather for %s.", location["name"])

            frame = fetch_weather(
                location,
                fetched_at=collection_time,
            )
            all_frames.append(frame)

            logger.info(
                "%s: fetched %s records.",
                location["name"],
                len(frame),
            )

        except (
            requests.RequestException,
            ValueError,
            KeyError,
            TypeError,
        ):
            failed_locations.append(location["name"])
            logger.exception(
                "Weather collection failed for %s.",
                location["name"],
            )

    if not all_frames:
        logger.error("No weather data collected; database was not updated.")
        return 0

    combined = pd.concat(all_frames, ignore_index=True)

    try:
        inserted = upsert_weather_records(combined)
    except Exception:
        logger.exception("Failed to upsert weather records into MySQL.")
        raise

    thunderstorm_available = combined[
        "thunderstorm_probability_pct"
    ].notna().sum()

    logger.info("Open-Meteo collection completed.")
    logger.info("Collection timestamp: %s UTC", collection_time.isoformat(sep=" "))
    logger.info("Successful locations: %s", len(all_frames))
    logger.info("Failed locations: %s", len(failed_locations))
    logger.info("Total records fetched: %s", len(combined))
    logger.info("New records inserted: %s", inserted)
    logger.info(
        "Thunderstorm probability available: %s/%s records.",
        thunderstorm_available,
        len(combined),
    )

    if failed_locations:
        logger.warning(
            "Locations requiring attention: %s",
            ", ".join(failed_locations),
        )

    return int(inserted or 0)


# Compatibility with the existing scheduler import.
def collect_all_locations() -> int:
    return collect_weather()


if __name__ == "__main__":
    collect_weather()
