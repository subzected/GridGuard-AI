
import pandas as pd


# ============================================================
# REQUIRED WEATHER COLUMNS
# ============================================================

WEATHER_REQUIRED = [
    "location_id",
    "snapshot_time",
    "temp_c",
    "humidity_pct",
    "precipitation_probability_pct",
    "precipitation_mm",
    "wind_speed_kmh",
    "wind_gusts_kmh",
    "pressure_hpa",
]


# Optional fields.
# cloud_cover_pct is optional because it does not exist
# in the confirmed MySQL weather_records schema.

WEATHER_OPTIONAL = [
    "cloud_cover_pct",
    "rain_mm",
    "wind_direction_deg",
    "thunderstorm_probability_pct",
    "latitude",
    "longitude",
    "forecast_valid_time",
    "fetched_at",
    "location_name",
    "weather_code",
    "source",
    "temporal_resolution",
]


# ============================================================
# REQUIRED OUTAGE EVENT COLUMNS
# ============================================================

EVENT_REQUIRED = [
    "location_id",
    "start_time",
    "end_time",
    "source",
    "verified",
]


# ============================================================
# REQUIRED COVERAGE COLUMNS
# ============================================================

COVERAGE_REQUIRED = [
    "location_id",
    "coverage_start",
    "coverage_end",
    "source",
]


# ============================================================
# COLUMN VALIDATION
# ============================================================

def validate_columns(df, required, name):

    missing = set(required) - set(df.columns)

    if missing:

        raise ValueError(
            f"{name}: missing columns {sorted(missing)}"
        )


# ============================================================
# TIMESTAMP PARSING
# ============================================================

def parse_timestamps(df, columns):

    for column in columns:

        if column in df.columns:

            df[column] = pd.to_datetime(
                df[column],
                utc=True,
                errors="coerce",
            )

    return df


# ============================================================
# WEATHER VALIDATION
# ============================================================

def validate_weather(df):

    validate_columns(
        df,
        WEATHER_REQUIRED,
        "Weather",
    )

    df = df.copy()

    # Parse timestamp fields.
    timestamp_columns = [
        "snapshot_time",
        "forecast_valid_time",
        "fetched_at",
    ]

    df = parse_timestamps(
        df,
        timestamp_columns,
    )

    if df["snapshot_time"].isna().any():

        raise ValueError(
            "Invalid weather snapshot timestamps."
        )

    if df["location_id"].isna().any():

        raise ValueError(
            "Weather location_id cannot be missing."
        )

    # Normalize location IDs.
    df["location_id"] = (
        df["location_id"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    # Convert numeric columns.
    numeric_columns = [
        "temp_c",
        "humidity_pct",
        "precipitation_probability_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "wind_gusts_kmh",
        "pressure_hpa",
        "cloud_cover_pct",
        "rain_mm",
        "wind_direction_deg",
        "thunderstorm_probability_pct",
        "latitude",
        "longitude",
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Check duplicate snapshots.
    if df.duplicated(
        [
            "location_id",
            "snapshot_time",
        ]
    ).any():

        raise ValueError(
            "Duplicate weather snapshots detected."
        )

    # Physical range validation.
    ranges = {
        "humidity_pct": (0, 100),

        "precipitation_probability_pct": (
            0,
            100,
        ),

        "precipitation_mm": (
            0,
            None,
        ),

        "rain_mm": (
            0,
            None,
        ),

        "wind_speed_kmh": (
            0,
            None,
        ),

        "wind_gusts_kmh": (
            0,
            None,
        ),

        "cloud_cover_pct": (
            0,
            100,
        ),

        "wind_direction_deg": (
            0,
            360,
        ),

        "thunderstorm_probability_pct": (
            0,
            100,
        ),

        "pressure_hpa": (
            800,
            1100,
        ),
    }

    for column, (low, high) in ranges.items():

        if column not in df.columns:
            continue

        values = df[column]

        invalid = values < low

        if high is not None:

            invalid |= values > high

        if invalid.fillna(False).any():

            raise ValueError(
                f"Out-of-range values in {column}"
            )

    return df


# ============================================================
# OUTAGE EVENT VALIDATION
# ============================================================

def validate_events(df):

    validate_columns(
        df,
        EVENT_REQUIRED,
        "Events",
    )

    df = parse_timestamps(
        df.copy(),
        [
            "start_time",
            "end_time",
        ],
    )

    if df["start_time"].isna().any():

        raise ValueError(
            "Event start_time is required."
        )

    # Normalize verified values.
    df["verified"] = (
        df["verified"]
        .astype(str)
        .str.lower()
        .isin([
            "true",
            "1",
            "yes",
        ])
    )

    # Check event intervals.
    invalid = (
        df["end_time"].notna()
        & (
            df["end_time"] < df["start_time"]
        )
    )

    if invalid.any():

        raise ValueError(
            "Event end_time precedes start_time."
        )

    # Only verified events are retained.
    return df[
        df["verified"]
    ].copy()


# ============================================================
# COVERAGE VALIDATION
# ============================================================

def validate_coverage(df):

    validate_columns(
        df,
        COVERAGE_REQUIRED,
        "Coverage",
    )

    df = parse_timestamps(
        df.copy(),
        [
            "coverage_start",
            "coverage_end",
        ],
    )

    if df[
        [
            "coverage_start",
            "coverage_end",
        ]
    ].isna().any().any():

        raise ValueError(
            "Invalid coverage timestamps."
        )

    if (
        df["coverage_end"]
        <= df["coverage_start"]
    ).any():

        raise ValueError(
            "Invalid coverage interval."
        )

    return df
