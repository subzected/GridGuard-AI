# USED
from pathlib import Path
import json
import time

import pandas as pd
import requests


BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

START_DATE = "2010-01-01"
END_DATE = "2025-12-31"

OUTPUT_DIR = Path("data/raw/open_meteo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOCATIONS = [
    {"location_id": "KOLKATA", "name": "Kolkata",
     "latitude": 22.5726, "longitude": 88.3639},
    {"location_id": "HOWRAH", "name": "Howrah",
     "latitude": 22.5958, "longitude": 88.2636},
    {"location_id": "ASANSOL", "name": "Asansol",
     "latitude": 23.6739, "longitude": 86.9524},
    {"location_id": "DURGAPUR", "name": "Durgapur",
     "latitude": 23.5204, "longitude": 87.3119},
    {"location_id": "HALDIA", "name": "Haldia",
     "latitude": 22.0667, "longitude": 88.0698},
    {"location_id": "SILIGURI", "name": "Siliguri",
     "latitude": 26.7271, "longitude": 88.3953},
    {"location_id": "DARJEELING","name": "Darjeeling",
    "latitude": 27.00,"longitude": 88.25},
]

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_mean",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "weather_code",
]


def fetch_location(location, max_retries=5):

    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "Asia/Kolkata",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "models": "era5",
    }

    session = requests.Session()

    for attempt in range(max_retries):

        try:
            response = session.get(
                BASE_URL,
                params=params,
                timeout=180
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after and retry_after.isdigit():
                    wait_time = int(retry_after)
                else:
                    wait_time = min(60 * (2 ** attempt), 300)

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)
                continue

            response.raise_for_status()

            payload = response.json()

            if payload.get("error"):
                raise RuntimeError(
                    payload.get("reason", "API error")
                )

            daily = payload.get("daily")

            if not daily or "time" not in daily:
                raise ValueError("No daily data returned")

            df = pd.DataFrame(daily)

            df["location_id"] = location["location_id"]
            df["location_name"] = location["name"]
            df["latitude"] = payload["latitude"]
            df["longitude"] = payload["longitude"]
            df["source"] = "Open-Meteo ERA5"

            return df, payload

        except requests.exceptions.HTTPError as exc:
            status = (
                exc.response.status_code
                if exc.response is not None else None
            )

            if status in (429, 500, 502, 503, 504):
                wait_time = min(60 * (2 ** attempt), 300)
                print(
                    f"HTTP {status}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
                continue

            raise

        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError
        ) as exc:
            wait_time = min(30 * (2 ** attempt), 240)

            print(
                f"Connection issue: {exc}. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        f"Failed to fetch {location['name']} after "
        f"{max_retries} attempts"
    )


def existing_file_is_valid(path):

    if not path.exists():
        return False

    try:
        df = pd.read_csv(path)

        return (
            not df.empty
            and "time" in df.columns
            and len(df) > 0
        )

    except Exception:
        return False


def main():

    successful = []
    failed = []

    for location in LOCATIONS:

        location_id = location["location_id"]

        csv_path = OUTPUT_DIR / f"{location_id}.csv"
        json_path = OUTPUT_DIR / f"{location_id}.json"

        print(f"\nProcessing: {location['name']}")

        # Resume: do not download existing valid files again.
        if existing_file_is_valid(csv_path):
            print("Already collected. Skipping API request.")

            successful.append(location_id)
            continue

        try:
            df, payload = fetch_location(location)

            # Save the JSON response.
            json_path.write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8"
            )

            # Save the CSV response.
            df.to_csv(csv_path, index=False)

            successful.append(location_id)

            print(f"Saved {len(df)} records.")

        except Exception as exc:
            failed.append({
                "location_id": location_id,
                "location": location["name"],
                "error": str(exc)
            })

            print(f"FAILED: {exc}")

        # Avoid sending requests too quickly.
        time.sleep(15)

    # Rebuild combined CSV from all valid location files.
    all_frames = []

    for location in LOCATIONS:

        csv_path = (
            OUTPUT_DIR / f"{location['location_id']}.csv"
        )

        if existing_file_is_valid(csv_path):
            all_frames.append(pd.read_csv(csv_path))

    if all_frames:

        combined = pd.concat(
            all_frames,
            ignore_index=True
        )

        combined.to_csv(
            OUTPUT_DIR / "openmeteo_historical_all.csv",
            index=False
        )

        print("\nCombined dataset generated.")
        print("Total records:", len(combined))
        print("Locations:", combined["location_id"].nunique())

    if failed:

        pd.DataFrame(failed).to_csv(
            OUTPUT_DIR / "collection_errors.csv",
            index=False
        )

    print("\n========== COLLECTION REPORT ==========")
    print("Successful:", len(successful))
    print("Failed:", len(failed))
    print("Successful locations:", successful)

    if failed:
        print("Check collection_errors.csv for failures.")


if __name__ == "__main__":
    main()
