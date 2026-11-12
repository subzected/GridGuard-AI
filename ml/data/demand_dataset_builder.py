"""Build a daily West Bengal electricity-demand + historical-weather dataset.

Inputs:
  data/processed/electricity_demand_features.csv
  data/processed/imd_historical_features.csv
  data/processed/openmeteo_historical_features.csv

Weather is aggregated across available West Bengal locations separately by
source. For each weather feature, median and maximum are retained. These are
descriptive spatial summaries, not population- or load-weighted estimates.

The historical weather inputs contain causal rolling features: each date's
weather features describe observations strictly before that date. They do not
represent a same-day weather forecast.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "electricity"

DEMAND_PATH = PROCESSED_DIR / "electricity_demand_features.csv"
IMD_PATH = PROCESSED_DIR / "imd_historical_features.csv"
OPEN_METEO_PATH = PROCESSED_DIR / "openmeteo_historical_features.csv"

OUTPUT_PATH = PROCESSED_DIR / "west_bengal_demand_weather_dataset.csv"
REPORT_PATH = REPORT_DIR / "demand_weather_fusion_report.json"

DATE_COLUMN = "date"
WEATHER_DATE_COLUMN = "snapshot_date"
TARGET_COLUMNS = ["energy_MU", "peak_demand_MW"]
EXCLUDE_EXACT = {
    "weather_source", "source", "source_name", "invalid_date_flag",
    "duplicate_key_flag", "invalid_temperature_flag",
    "invalid_precipitation_flag", "invalid_humidity_flag",
    "invalid_wind_flag", "invalid_pressure_flag",
    "missing_temperature_flag", "missing_precipitation_flag",
    "temperature_outlier_flag", "precipitation_outlier_flag",
    "temperature_series_duplicate_flag",
    "location_temperature_review_flag",
    "location_precipitation_review_flag",
    "data_quality_issue_flag", "review_flag", "quality_issue_flag",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _load_demand() -> pd.DataFrame:
    if not DEMAND_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DEMAND_PATH}. Run demand feature engineering first."
        )
    df = pd.read_csv(DEMAND_PATH, parse_dates=[DATE_COLUMN])
    required = [DATE_COLUMN, *TARGET_COLUMNS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Demand feature file missing columns: {missing}")
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dt.normalize()
    if df[DATE_COLUMN].isna().any():
        raise ValueError("Demand feature file contains invalid dates.")
    if df[DATE_COLUMN].duplicated().any():
        raise ValueError("Demand feature file contains duplicate dates.")
    return df.sort_values(DATE_COLUMN).reset_index(drop=True)


def _aggregate_weather(path: Path, source_prefix: str) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing weather feature file: {path}")

    df = pd.read_csv(path)
    required = [WEATHER_DATE_COLUMN, "location_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")

    df[WEATHER_DATE_COLUMN] = pd.to_datetime(
        df[WEATHER_DATE_COLUMN], errors="coerce"
    ).dt.normalize()
    df["location_id"] = df["location_id"].astype("string").str.strip().str.lower()
    df = df.dropna(subset=[WEATHER_DATE_COLUMN, "location_id"]).copy()

    if df.duplicated([WEATHER_DATE_COLUMN, "location_id"]).any():
        raise ValueError(
            f"{path.name} has duplicate date/location rows; resolve before fusion."
        )

    # Use only this source's engineered weather predictors. QC/provenance flags
    # are retained in the report, never included as model features.
    candidate_cols = [
        c for c in df.columns
        if c.startswith(source_prefix + "_")
        and c not in EXCLUDE_EXACT
    ]
    numeric_cols = []
    for col in candidate_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].notna().any():
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError(f"No usable numeric {source_prefix} features in {path.name}.")

    # Median is robust to spatial outliers; max preserves local extremes.
    # Counts expose how many stations contributed to each daily summary.
    grouped = df.groupby(WEATHER_DATE_COLUMN, sort=True)
    median = grouped[numeric_cols].median().add_suffix("_spatial_median")
    maximum = grouped[numeric_cols].max().add_suffix("_spatial_max")
    valid_counts = grouped[numeric_cols].count().min(axis=1).rename(
        f"{source_prefix}_locations_with_any_weather_feature"
    )
    location_counts = grouped["location_id"].nunique().rename(
        f"{source_prefix}_location_count"
    )

    result = pd.concat([median, maximum, valid_counts, location_counts], axis=1)
    result.index.name = DATE_COLUMN
    result = result.reset_index()

    report = {
        "file": path.name,
        "rows": int(len(df)),
        "locations": int(df["location_id"].nunique()),
        "date_start": df[WEATHER_DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "date_end": df[WEATHER_DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "input_numeric_features": numeric_cols,
        "aggregated_feature_count": int(len(result.columns) - 1),
        "aggregation": (
            "Daily spatial median and maximum across available locations; "
            "location counts included. Not population- or load-weighted."
        ),
    }
    return result, report


def build_demand_weather_dataset() -> pd.DataFrame:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    demand = _load_demand()
    imd, imd_report = _aggregate_weather(IMD_PATH, "imd")
    open_meteo, om_report = _aggregate_weather(OPEN_METEO_PATH, "open_meteo")

    # One row per date, preserving demand dates and missing target dates.
    dataset = demand.merge(imd, on=DATE_COLUMN, how="left", validate="one_to_one")
    dataset = dataset.merge(
        open_meteo, on=DATE_COLUMN, how="left", validate="one_to_one"
    )
    dataset = dataset.sort_values(DATE_COLUMN).reset_index(drop=True)

    weather_feature_cols = [
        c for c in dataset.columns
        if c.startswith(("imd_", "open_meteo_"))
        and c not in TARGET_COLUMNS
    ]
    report = {
        "geography": "West Bengal state-level demand with spatially aggregated weather",
        "frequency": "daily",
        "demand_rows": int(len(demand)),
        "output_rows": int(len(dataset)),
        "date_start": dataset[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "date_end": dataset[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "targets": TARGET_COLUMNS,
        "imd": imd_report,
        "open_meteo": om_report,
        "weather_feature_columns": weather_feature_cols,
        "missing_by_column": {
            c: int(n) for c, n in dataset.isna().sum().items()
        },
        "temporal_policy": (
            "Demand-derived features for date D use demand strictly before D. "
            "Input historical weather features for date D also use weather "
            "observations strictly before D. This dataset therefore represents "
            "a historical-information baseline, not a same-day weather-forecast "
            "model. Same-day forecast features require forecast snapshots joined "
            "by forecast valid date and retrieval-time availability."
        ),
        "spatial_policy": (
            "IMD and Open-Meteo are aggregated separately by date using spatial "
            "median and maximum across available locations. These summaries are "
            "not load-weighted and should be interpreted as regional indicators."
        ),
        "join_policy": (
            "Left join from daily electricity demand calendar to source-specific "
            "daily weather summaries. Demand rows are never dropped because "
            "weather features are unavailable."
        ),
    }

    dataset.to_csv(OUTPUT_PATH, index=False)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Weather-demand dataset fusion completed.")
    logger.info("Rows: %s | Columns: %s", len(dataset), len(dataset.columns))
    logger.info("Date range: %s to %s", report["date_start"], report["date_end"])
    logger.info("IMD locations: %s", imd_report["locations"])
    logger.info("Open-Meteo locations: %s", om_report["locations"])
    logger.info("Weather features: %s", len(weather_feature_cols))
    logger.info("Saved dataset: %s", OUTPUT_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    return dataset


if __name__ == "__main__":
    build_demand_weather_dataset()
