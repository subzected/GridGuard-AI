import pandas as pd

from sqlalchemy import select, desc

from src.database.connection import SessionLocal
from src.database.models import WeatherRecord


FIELDS = [
    "location_id",
    "location_name",
    "latitude",
    "longitude",
    "valid_time",
    "fetched_at",
    "temperature_c",
    "relative_humidity_pct",
    "precipitation_mm",
    "rain_mm",
    "wind_speed_kmh",
    "wind_gusts_kmh",
    "wind_direction_deg",
    "pressure_hpa",
    "precipitation_probability_pct",
    "thunderstorm_probability_pct",
    "weather_code",
    "source",
    "temporal_resolution",
]


def upsert_weather_records(frame: pd.DataFrame) -> int:
    """
    Insert weather forecast snapshots.

    Each unique combination of location, forecast validity time,
    fetch time and source is preserved.

    Returns the number of newly inserted records.
    """

    if frame.empty:
        return 0

    clean = frame.copy()

    for col in FIELDS:
        if col not in clean.columns:
            clean[col] = None

    clean = clean[FIELDS].where(pd.notna(clean[FIELDS]), None)

    rows = clean.to_dict("records")

    inserted = 0

    with SessionLocal() as session:

        for row in rows:

            existing = session.scalar(
                select(WeatherRecord).where(
                    WeatherRecord.location_id == row["location_id"],
                    WeatherRecord.valid_time == row["valid_time"],
                    WeatherRecord.fetched_at == row["fetched_at"],
                    WeatherRecord.source == (row["source"] or "unknown"),
                )
            )

            if existing:
                # Same forecast snapshot already exists.
                continue

            session.add(WeatherRecord(**row))
            inserted += 1

        session.commit()

    return inserted


def latest_weather() -> pd.DataFrame:

    with SessionLocal() as session:

        rows = session.scalars(
            select(WeatherRecord)
            .order_by(desc(WeatherRecord.fetched_at))
            .limit(1000)
        ).all()

        return pd.DataFrame([
            {column: getattr(row, column) for column in FIELDS}
            for row in rows
        ])


def weather_history(
    location_name: str,
    limit: int = 500
) -> pd.DataFrame:

    with SessionLocal() as session:

        rows = session.scalars(
            select(WeatherRecord)
            .where(
                WeatherRecord.location_name == location_name
            )
            .order_by(
                desc(WeatherRecord.fetched_at)
            )
            .limit(limit)
        ).all()

        data = [
            {column: getattr(row, column) for column in FIELDS}
            for row in rows
        ]

    frame = pd.DataFrame(data)

    if not frame.empty:
        frame = frame.sort_values(
            ["fetched_at", "valid_time"]
        ).reset_index(drop=True)

    return frame