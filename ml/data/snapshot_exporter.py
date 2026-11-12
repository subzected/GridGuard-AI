
"""
GridGuard AI - Weather Snapshot Exporter

Reads weather_records from MySQL and exports decision-time weather snapshots.

Timestamp contract:
    fetched_at -> snapshot_time
    valid_time -> forecast_valid_time

One row is exported per location and fetch batch. The exporter does not
manufacture 15-minute snapshots from hourly forecasts.
"""

from pathlib import Path
import json
import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

OUTPUT_PATH = ROOT / "data" / "raw" / "weather_snapshots.csv"
REPORT_PATH = (
    ROOT / "reports" / "weather_snapshots"
    / "snapshot_export_report.json"
)

FETCH_BATCH_ROUNDING = "1min"
MAX_VALID_TIME_DISTANCE_MINUTES = 90

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("snapshot_exporter")


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_database_engine():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing from the project .env file."
        )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
    )

    return engine
    

# ============================================================
# LOAD WEATHER RECORDS
# ============================================================

def load_weather_records():

    query = text("""
        SELECT
            id,
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
        WHERE fetched_at IS NOT NULL
          AND valid_time IS NOT NULL
        ORDER BY location_id, fetched_at, valid_time
    """)

    engine = get_database_engine()

    try:
        with engine.connect() as connection:
            records = pd.read_sql(query, connection)

    finally:
        engine.dispose()

    if records.empty:
        raise ValueError(
            "No records found in weather_records."
        )

    return records


# ============================================================
# NORMALIZE DATABASE COLUMNS
# ============================================================

def normalize_columns(df):

    result = pd.DataFrame(index=df.index)

    result["location_id"] = (
        df["location_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    result["location_name"] = df["location_name"]

    result["latitude"] = df["latitude"]
    result["longitude"] = df["longitude"]

    # Timestamp mapping
    result["snapshot_time"] = df["fetched_at"]
    result["forecast_valid_time"] = df["valid_time"]
    result["fetched_at"] = df["fetched_at"]

    # Weather variables
    result["temp_c"] = df["temperature_c"]

    result["humidity_pct"] = (
        df["relative_humidity_pct"]
    )

    result["precipitation_mm"] = (
        df["precipitation_mm"]
    )

    result["rain_mm"] = df["rain_mm"]

    result["wind_speed_kmh"] = (
        df["wind_speed_kmh"]
    )

    result["wind_gusts_kmh"] = (
        df["wind_gusts_kmh"]
    )

    result["wind_direction_deg"] = (
        df["wind_direction_deg"]
    )

    result["pressure_hpa"] = (
        df["pressure_hpa"]
    )

    result["precipitation_probability_pct"] = (
        df["precipitation_probability_pct"]
    )

    result["thunderstorm_probability_pct"] = (
        df["thunderstorm_probability_pct"]
    )

    result["weather_code"] = df["weather_code"]

    result["source"] = df["source"]

    result["temporal_resolution"] = (
        df["temporal_resolution"]
    )

    return result


# ============================================================
# BUILD WEATHER SNAPSHOTS
# ============================================================

def build_snapshots(raw):

    data = raw.copy()

    data["location_id"] = (
        data["location_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    data["fetched_at"] = pd.to_datetime(
        data["fetched_at"],
        utc=True,
        errors="coerce",
    )

    data["valid_time"] = pd.to_datetime(
        data["valid_time"],
        utc=True,
        errors="coerce",
    )

    data = data.dropna(
        subset=[
            "location_id",
            "fetched_at",
            "valid_time",
        ]
    ).copy()

    if data.empty:
        raise ValueError(
            "No rows have valid location_id, fetched_at, and valid_time."
        )

    # Group records belonging to the same fetch cycle.
    data["_fetch_batch"] = (
        data["fetched_at"].dt.floor(
            FETCH_BATCH_ROUNDING
        )
    )

    # Calculate distance between forecast time and fetch time.
    data["_valid_distance_minutes"] = (
        data["valid_time"] - data["fetched_at"]
    ).abs().dt.total_seconds() / 60

    # Keep forecast records close to the retrieval time.
    candidates = data[
        data["_valid_distance_minutes"]
        <= MAX_VALID_TIME_DISTANCE_MINUTES
    ].copy()

    if candidates.empty:
        raise ValueError(
            "No forecast valid_time is within the configured "
            "90-minute window of fetched_at. Check timezone semantics."
        )

    # Select the forecast whose valid_time is closest
    # to the actual retrieval timestamp.
    candidates = candidates.sort_values(
        [
            "location_id",
            "_fetch_batch",
            "_valid_distance_minutes",
            "valid_time",
        ]
    )

    selected = candidates.drop_duplicates(
        [
            "location_id",
            "_fetch_batch",
        ],
        keep="first",
    ).copy()

    # Use the latest actual fetch timestamp in each batch.
    batch_times = (
        data.groupby(
            ["location_id", "_fetch_batch"]
        )["fetched_at"]
        .max()
        .rename("_batch_snapshot_time")
        .reset_index()
    )

    selected = selected.merge(
        batch_times,
        on=[
            "location_id",
            "_fetch_batch",
        ],
        how="left",
        validate="one_to_one",
    )

    selected["fetched_at"] = (
        selected["_batch_snapshot_time"]
    )

    snapshots = normalize_columns(selected)

    # Verify uniqueness.
    if snapshots.duplicated(
        ["location_id", "snapshot_time"]
    ).any():

        raise ValueError(
            "Duplicate (location_id, snapshot_time) keys were generated."
        )

    snapshots = snapshots.sort_values(
        [
            "location_id",
            "snapshot_time",
        ]
    ).reset_index(drop=True)

    # Convert timestamps to ISO-8601 UTC strings.
    for column in [
        "snapshot_time",
        "forecast_valid_time",
        "fetched_at",
    ]:

        snapshots[column] = pd.to_datetime(
            snapshots[column],
            utc=True,
        ).dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    return snapshots


# ============================================================
# GENERATE EXPORT REPORT
# ============================================================

def create_report(raw, snapshots):

    snapshot_times = pd.to_datetime(
        snapshots["snapshot_time"],
        utc=True,
    )

    valid_times = pd.to_datetime(
        snapshots["forecast_valid_time"],
        utc=True,
    )

    offset_minutes = (
        valid_times - snapshot_times
    ).dt.total_seconds() / 60

    weather_columns = [
        "temp_c",
        "humidity_pct",
        "precipitation_mm",
        "rain_mm",
        "wind_speed_kmh",
        "wind_gusts_kmh",
        "wind_direction_deg",
        "pressure_hpa",
        "precipitation_probability_pct",
        "thunderstorm_probability_pct",
    ]

    return {
        "project": "GridGuard AI",

        "report_type": "Weather Snapshot Export",

        "source_table": "weather_records",

        "raw_database_rows": int(len(raw)),

        "exported_snapshot_rows": int(
            len(snapshots)
        ),

        "locations": int(
            snapshots["location_id"].nunique()
        ),

        "location_ids": sorted(
            snapshots["location_id"]
            .dropna()
            .unique()
            .tolist()
        ),

        "first_snapshot_time": (
            snapshot_times.min().isoformat()
        ),

        "last_snapshot_time": (
            snapshot_times.max().isoformat()
        ),

        "duplicate_snapshot_keys": int(
            snapshots.duplicated(
                [
                    "location_id",
                    "snapshot_time",
                ]
            ).sum()
        ),

        "forecast_valid_time_offset_minutes": {
            "minimum": float(
                offset_minutes.min()
            ),
            "median": float(
                offset_minutes.median()
            ),
            "maximum": float(
                offset_minutes.max()
            ),
        },

        "missing_percentage": {
            col: round(
                float(
                    snapshots[col].isna().mean() * 100
                ),
                2,
            )
            for col in weather_columns
        },

        "source_counts": (
            snapshots["source"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .to_dict()
        ),

        "temporal_resolution_counts": (
            snapshots["temporal_resolution"]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
            .to_dict()
        ),

        "timestamp_policy": {
            "snapshot_time": "fetched_at",
            "forecast_valid_time": "valid_time",
            "fetch_batch_rounding": (
                FETCH_BATCH_ROUNDING
            ),
            "max_valid_time_distance_minutes": (
                MAX_VALID_TIME_DISTANCE_MINUTES
            ),
        },

        "notes": [
            "One forecast row selected per location and fetch batch.",
            "No artificial 15-minute snapshots are created.",
            "Cloud cover is not present in the confirmed MySQL schema.",
            "Missing weather values are preserved; no imputation is performed.",
        ],
    }


# ============================================================
# MAIN EXPORT FUNCTION
# ============================================================

def export_weather_snapshots():

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Reading weather_records from MySQL..."
    )

    raw = load_weather_records()

    logger.info(
        "Loaded %s database rows.",
        len(raw),
    )

    snapshots = build_snapshots(raw)

    # Export CSV.
    snapshots.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # Export report.
    report = create_report(
        raw,
        snapshots,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    print("\n========== SNAPSHOT EXPORT ==========")

    print(f"Database rows:  {len(raw):,}")

    print(
        f"Snapshot rows:  {len(snapshots):,}"
    )

    print(
        f"Locations:      {snapshots['location_id'].nunique()}"
    )

    print(
        f"First snapshot: {snapshots['snapshot_time'].min()}"
    )

    print(
        f"Last snapshot:  {snapshots['snapshot_time'].max()}"
    )

    print(f"CSV:            {OUTPUT_PATH}")

    print(f"Report:         {REPORT_PATH}")

    print("=====================================")

    return snapshots


if __name__ == "__main__":

    export_weather_snapshots()
