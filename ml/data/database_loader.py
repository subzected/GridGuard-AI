
import os

import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv


load_dotenv()


def get_database_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")

    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600
    )


def load_weather_records():
    engine = get_database_engine()

    query = """
        SELECT
            location_id,
            location_name,
            latitude,
            longitude,
            valid_time,
            fetched_at,
            temperature_c,
            relative_humidity_pct,
            precipitation_mm,
            rain_mm,
            wind_speed_kmh,
            wind_gusts_kmh,
            wind_direction_deg,
            pressure_hpa,
            precipitation_probability_pct,
            thunderstorm_probability_pct,
            weather_code,
            source,
            temporal_resolution
        FROM weather_records
        ORDER BY location_id, fetched_at, valid_time
    """

    with engine.connect() as connection:
        df = pd.read_sql(query, connection)

    return df
