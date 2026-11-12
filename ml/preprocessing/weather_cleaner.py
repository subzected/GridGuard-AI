from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

IMD_PATH = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OPENMETEO_PATH = ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"
SOURCE_COMPARISON_PATH = ROOT / "reports/weather_quality/source_comparison.csv"

INTERIM_DIR = ROOT / "data/interim"
PROCESSED_DIR = ROOT / "data/processed"
REPORT_DIR = ROOT / "reports/weather_cleaning"


# Broad screening limits, not official meteorological standards.
# Observations outside these limits are flagged, never deleted.
VALID_RANGES = {
    "temperature_c": (-20, 60),
    "precipitation_mm": (0, 1000),
    "relative_humidity_pct": (0, 100),
    "wind_speed_ms": (0, 150),
    "pressure_hpa": (800, 1100),
}


def first_available_column(df, candidates):
    """Return the first available column from a list."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def detect_iqr_outliers(series):
    """
    Flag statistical outliers using the IQR method.
    These are review indicators, not proof of invalid observations.
    """
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    if len(valid) < 10:
        return pd.Series(False, index=series.index)

    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1

    if pd.isna(iqr) or iqr == 0:
        return pd.Series(False, index=series.index)

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return ((values < lower) | (values > upper)).fillna(False)


def detect_duplicate_temperature_locations(
    df,
    temperature_column,
    min_common_days=365,
):
    """
    Identify locations with identical temperature values on all
    jointly available dates. No observations are removed.
    """
    if temperature_column is None:
        return set()

    required = ["location_id", "snapshot_date", temperature_column]
    data = df[required].dropna(
        subset=["location_id", "snapshot_date", temperature_column]
    ).copy()

    # Deduplicate only the temporary comparison table.
    data = data.drop_duplicates(
        subset=["location_id", "snapshot_date"],
        keep="first",
    )

    if data.empty:
        return set()

    pivot = data.pivot(
        index="snapshot_date",
        columns="location_id",
        values=temperature_column,
    )

    locations = list(pivot.columns)
    duplicate_locations = set()

    for i in range(len(locations)):
        for j in range(i + 1, len(locations)):
            location_a = locations[i]
            location_b = locations[j]
            pair = pivot[[location_a, location_b]].dropna()

            if len(pair) < min_common_days:
                continue

            identical = np.allclose(
                pair[location_a].to_numpy(dtype=float),
                pair[location_b].to_numpy(dtype=float),
                atol=0.001,
                rtol=0,
            )

            if identical:
                duplicate_locations.update([location_a, location_b])

    return duplicate_locations


def apply_source_comparison_flags(df):
    """
    Attach location-level IMD/Open-Meteo comparison results.
    These flags describe source agreement, not which source is wrong.
    """
    df["location_temperature_comparison_quality"] = "NOT_COMPARED"
    df["location_precipitation_comparison_quality"] = "NOT_COMPARED"

    if not SOURCE_COMPARISON_PATH.exists():
        print(
            "Warning: source_comparison.csv not found. "
            "Location-level comparison flags were not applied."
        )
        return df

    comparison = pd.read_csv(SOURCE_COMPARISON_PATH)

    required = {
        "location_id",
        "temperature_quality",
        "precipitation_quality",
    }
    missing = required.difference(comparison.columns)

    if missing:
        print(
            "Warning: source_comparison.csv is missing columns: "
            f"{sorted(missing)}. Comparison flags were not applied."
        )
        return df

    comparison["location_id"] = (
        comparison["location_id"].astype(str).str.strip().str.lower()
    )
    comparison = comparison.drop_duplicates(
        subset=["location_id"],
        keep="first",
    )

    temperature_map = comparison.set_index(
        "location_id"
    )["temperature_quality"].to_dict()

    precipitation_map = comparison.set_index(
        "location_id"
    )["precipitation_quality"].to_dict()

    df["location_temperature_comparison_quality"] = (
        df["location_id"].map(temperature_map).fillna("NOT_COMPARED")
    )
    df["location_precipitation_comparison_quality"] = (
        df["location_id"].map(precipitation_map).fillna("NOT_COMPARED")
    )

    return df


def clean_weather_dataset(path, source_name, date_column):
    """Standardize and quality-flag one weather source."""

    print(f"\nCleaning {source_name} dataset...")

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    original_records = len(df)

    required_columns = {"location_id", date_column}
    missing_required = required_columns.difference(df.columns)
    if missing_required:
        raise ValueError(
            f"{source_name} dataset is missing required columns: "
            f"{sorted(missing_required)}"
        )

    # 1. Normalize location identifiers.
    df["location_id"] = (
        df["location_id"].astype("string").str.strip().str.lower()
    )

    # 2. Normalize dates.
    df["snapshot_date"] = pd.to_datetime(
        df[date_column], errors="coerce"
    ).dt.normalize()
    df["invalid_date_flag"] = df["snapshot_date"].isna()

    # 3. Detect duplicate location-date keys.
    df["duplicate_key_flag"] = df.duplicated(
        subset=["location_id", "snapshot_date"],
        keep=False,
    )

    # 4. Identify source-specific weather columns.
    if source_name == "IMD":
        temperature_column = first_available_column(
            df, ["temperature_c"]
        )
        precipitation_column = first_available_column(
            df, ["precipitation_mm", "rain_mm", "rain"]
        )
        humidity_column = None
        wind_column = None
        pressure_column = None

    elif source_name == "OPEN_METEO":
        temperature_column = first_available_column(
            df, ["temperature_2m_mean"]
        )
        precipitation_column = first_available_column(
            df, ["precipitation_sum", "rain_sum"]
        )
        humidity_column = first_available_column(
            df, ["relative_humidity_2m_mean"]
        )
        wind_column = first_available_column(
            df, ["wind_speed_10m_mean"]
        )
        pressure_column = first_available_column(
            df, ["pressure_msl_mean"]
        )

    else:
        raise ValueError("source_name must be 'IMD' or 'OPEN_METEO'.")

    # 5. Standardize weather variables.
    def numeric_column(column):
        if column is None:
            return pd.Series(np.nan, index=df.index, dtype=float)
        return pd.to_numeric(df[column], errors="coerce")

    df["temperature_c"] = numeric_column(temperature_column)
    df["precipitation_mm"] = numeric_column(precipitation_column)
    df["relative_humidity_pct"] = numeric_column(humidity_column)
    df["wind_speed_ms"] = numeric_column(wind_column)
    df["pressure_hpa"] = numeric_column(pressure_column)

    # 6. Physical-range validation.
    df["invalid_temperature_flag"] = (
        (
            (df["temperature_c"] < VALID_RANGES["temperature_c"][0])
            | (df["temperature_c"] > VALID_RANGES["temperature_c"][1])
        ).fillna(False)
    )
    df["invalid_precipitation_flag"] = (
        (
            (df["precipitation_mm"] < VALID_RANGES["precipitation_mm"][0])
            | (df["precipitation_mm"] > VALID_RANGES["precipitation_mm"][1])
        ).fillna(False)
    )
    df["invalid_humidity_flag"] = (
        (
            (df["relative_humidity_pct"] < 0)
            | (df["relative_humidity_pct"] > 100)
        ).fillna(False)
    )
    df["invalid_wind_flag"] = (
        (
            (df["wind_speed_ms"] < VALID_RANGES["wind_speed_ms"][0])
            | (df["wind_speed_ms"] > VALID_RANGES["wind_speed_ms"][1])
        ).fillna(False)
    )
    df["invalid_pressure_flag"] = (
        (
            (df["pressure_hpa"] < VALID_RANGES["pressure_hpa"][0])
            | (df["pressure_hpa"] > VALID_RANGES["pressure_hpa"][1])
        ).fillna(False)
    )

    # 7. Missing required-variable flags.
    df["missing_temperature_flag"] = df["temperature_c"].isna()
    df["missing_precipitation_flag"] = df["precipitation_mm"].isna()

    # 8. Statistical outlier detection, separately by location.
    df["temperature_outlier_flag"] = False
    df["precipitation_outlier_flag"] = False

    for _, indices in df.groupby("location_id").groups.items():
        location_data = df.loc[indices]
        df.loc[indices, "temperature_outlier_flag"] = (
            detect_iqr_outliers(location_data["temperature_c"])
        )
        df.loc[indices, "precipitation_outlier_flag"] = (
            detect_iqr_outliers(location_data["precipitation_mm"])
        )

    # 9. Flag locations with duplicate temperature series.
    duplicate_locations = detect_duplicate_temperature_locations(
        df, "temperature_c"
    )
    df["temperature_series_duplicate_flag"] = (
        df["location_id"].isin(duplicate_locations)
    )

    # 10. Attach location-level source comparison flags.
    df = apply_source_comparison_flags(df)

    df["location_temperature_review_flag"] = (
        df["location_temperature_comparison_quality"].isin(
            ["REVIEW", "CRITICAL"]
        )
    )
    df["location_precipitation_review_flag"] = (
        df["location_precipitation_comparison_quality"].isin(
            ["REVIEW", "CRITICAL"]
        )
    )

    # 11. Separate actual data-quality issues from review indicators.
    # Missing optional IMD humidity/wind/pressure fields are not
    # considered errors because those variables are not supplied here.
    data_quality_columns = [
        "invalid_date_flag",
        "duplicate_key_flag",
        "missing_temperature_flag",
        "missing_precipitation_flag",
        "invalid_temperature_flag",
        "invalid_precipitation_flag",
        "invalid_humidity_flag",
        "invalid_wind_flag",
        "invalid_pressure_flag",
    ]
    df["data_quality_issue_flag"] = df[
        data_quality_columns
    ].any(axis=1)

    review_columns = [
        "temperature_outlier_flag",
        "precipitation_outlier_flag",
        "temperature_series_duplicate_flag",
        "location_temperature_review_flag",
        "location_precipitation_review_flag",
    ]
    df["review_flag"] = df[review_columns].any(axis=1)

    df["quality_issue_flag"] = (
        df["data_quality_issue_flag"] | df["review_flag"]
    )

    # 12. Source label.
    df["source_name"] = source_name

    # 13. Cleaning summary.
    summary = {
        "source": source_name,
        "original_records": int(original_records),
        "output_records": int(len(df)),
        "locations": int(df["location_id"].nunique()),
        "invalid_dates": int(df["invalid_date_flag"].sum()),
        "duplicate_location_date_keys": int(
            df["duplicate_key_flag"].sum()
        ),
        "missing_temperature": int(df["missing_temperature_flag"].sum()),
        "missing_precipitation": int(
            df["missing_precipitation_flag"].sum()
        ),
        "invalid_temperature": int(df["invalid_temperature_flag"].sum()),
        "invalid_precipitation": int(
            df["invalid_precipitation_flag"].sum()
        ),
        "invalid_humidity": int(df["invalid_humidity_flag"].sum()),
        "invalid_wind": int(df["invalid_wind_flag"].sum()),
        "invalid_pressure": int(df["invalid_pressure_flag"].sum()),
        "temperature_outliers": int(
            df["temperature_outlier_flag"].sum()
        ),
        "precipitation_outliers": int(
            df["precipitation_outlier_flag"].sum()
        ),
        "duplicate_temperature_locations": sorted(
            duplicate_locations
        ),
        "rows_with_data_quality_issues": int(
            df["data_quality_issue_flag"].sum()
        ),
        "rows_requiring_review": int(df["review_flag"].sum()),
        "rows_with_any_quality_flags": int(
            df["quality_issue_flag"].sum()
        ),
        "rows_removed": 0,
    }

    print(f"{source_name} records: {len(df)}")
    print(f"{source_name} locations: {summary['locations']}")
    print(
        f"{source_name} data-quality issues: "
        f"{summary['rows_with_data_quality_issues']}"
    )
    print(
        f"{source_name} rows requiring review: "
        f"{summary['rows_requiring_review']}"
    )

    return df, summary


def run_weather_cleaning():
    """Run the cleaning pipeline for IMD and Open-Meteo."""

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print(" GRIDGUARD AI - WEATHER DATA CLEANING")
    print("======================================")

    # 1. Clean IMD.
    imd, imd_summary = clean_weather_dataset(
        IMD_PATH,
        source_name="IMD",
        date_column="valid_time",
    )

    # 2. Clean Open-Meteo.
    openmeteo, om_summary = clean_weather_dataset(
        OPENMETEO_PATH,
        source_name="OPEN_METEO",
        date_column="time",
    )

    # 3. Save source-specific datasets.
    imd_output = INTERIM_DIR / "imd_cleaned.csv"
    om_output = INTERIM_DIR / "openmeteo_cleaned.csv"

    imd.to_csv(imd_output, index=False)
    openmeteo.to_csv(om_output, index=False)

    # 4. Create standardized long-format dataset.
    common_columns = [
        "snapshot_date",
        "location_id",
        "temperature_c",
        "precipitation_mm",
        "relative_humidity_pct",
        "wind_speed_ms",
        "pressure_hpa",
        "source_name",
        "data_quality_issue_flag",
        "review_flag",
        "quality_issue_flag",
        "temperature_outlier_flag",
        "precipitation_outlier_flag",
        "temperature_series_duplicate_flag",
        "location_temperature_review_flag",
        "location_precipitation_review_flag",
    ]

    # Verify that every requested column exists in both source datasets.
    for source_label, source_df in [
        ("IMD", imd),
        ("OPEN_METEO", openmeteo),
    ]:
        missing_columns = [
            column for column in common_columns
            if column not in source_df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{source_label} cleaned dataset is missing columns: "
                f"{missing_columns}"
            )

    combined = pd.concat(
        [
            imd[common_columns],
            openmeteo[common_columns],
        ],
        ignore_index=True,
    )

    combined = combined.sort_values(
        ["location_id", "snapshot_date", "source_name"]
    ).reset_index(drop=True)

    combined_output = PROCESSED_DIR / "weather_cleaned_long.csv"
    combined.to_csv(combined_output, index=False)

    # 5. Save cleaning report.
    report = {
        "project": "GridGuard AI",
        "report_type": "Weather Data Cleaning",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "imd": imd_summary,
        "openmeteo": om_summary,
        "combined": {
            "records": int(len(combined)),
            "locations": int(combined["location_id"].nunique()),
            "sources": combined["source_name"].unique().tolist(),
            "rows_with_data_quality_issues": int(
                combined["data_quality_issue_flag"].sum()
            ),
            "rows_requiring_review": int(combined["review_flag"].sum()),
            "rows_with_any_quality_flags": int(
                combined["quality_issue_flag"].sum()
            ),
        },
        "cleaning_policy": [
            "Raw datasets were not modified.",
            "No observations were automatically deleted.",
            "Missing values were not automatically imputed.",
            "IMD and Open-Meteo observations remain separate.",
            "Statistical outliers were flagged, not removed.",
            "Duplicate temperature series are treated as spatial redundancy.",
            "Source comparison flags describe location-level agreement, not individual source validity.",
            "Physical screening ranges are broad diagnostics, not official meteorological standards.",
        ],
    }

    report_path = REPORT_DIR / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    # 6. Final output.
    print("\n======================================")
    print(" WEATHER CLEANING COMPLETED")
    print("======================================")
    print("IMD cleaned:", imd_output)
    print("Open-Meteo cleaned:", om_output)
    print("Combined dataset:", combined_output)
    print("Cleaning report:", report_path)
    print("\nRecords retained:", len(combined))
    print(
        "Rows with data-quality issues:",
        int(combined["data_quality_issue_flag"].sum()),
    )
    print(
        "Rows requiring review:",
        int(combined["review_flag"].sum()),
    )
    print("Rows removed: 0")


if __name__ == "__main__":
    run_weather_cleaning()
