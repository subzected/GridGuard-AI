"""Causal daily demand feature engineering for GridGuard AI.

Input:
    data/processed/electricity_daily_cleaned.csv

Output:
    data/processed/electricity_demand_features.csv
    reports/electricity/demand_feature_report.json

All demand-derived predictors for date D use observations strictly before D.
Missing calendar dates are inserted as rows with missing targets so that lag
and rolling features represent calendar days rather than previous available rows.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "electricity_daily_cleaned.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "electricity"
OUTPUT_PATH = OUTPUT_DIR / "electricity_demand_features.csv"
REPORT_PATH = REPORT_DIR / "demand_feature_report.json"

DATE_COLUMN = "date"
TARGET_COLUMNS = ["energy_MU", "peak_demand_MW"]
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 30)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_input(df: pd.DataFrame) -> pd.DataFrame:
    required = [DATE_COLUMN, *TARGET_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Electricity dataset is missing columns: {missing}")

    result = df[required].copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN], errors="coerce")
    if result[DATE_COLUMN].isna().any():
        bad_count = int(result[DATE_COLUMN].isna().sum())
        raise ValueError(f"Electricity dataset contains {bad_count} invalid dates.")

    # The source is daily. Normalize timestamps to dates and reject collisions.
    result[DATE_COLUMN] = result[DATE_COLUMN].dt.normalize()
    for column in TARGET_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        if np.isinf(result[column].to_numpy(dtype=float, na_value=np.nan)).any():
            raise ValueError(f"{column} contains infinite values.")
        if (result[column].dropna() < 0).any():
            raise ValueError(f"{column} contains negative values.")

    if result[DATE_COLUMN].duplicated().any():
        examples = (
            result.loc[result[DATE_COLUMN].duplicated(keep=False), DATE_COLUMN]
            .dt.strftime("%Y-%m-%d").head(10).tolist()
        )
        raise ValueError(
            "Duplicate dates found in electricity dataset. "
            f"Resolve them in electricity_loader.py first. Examples: {examples}"
        )

    return result.sort_values(DATE_COLUMN).reset_index(drop=True)


def engineer_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and strictly-past demand features on a complete daily index."""
    data = _validate_input(df)

    full_dates = pd.date_range(
        start=data[DATE_COLUMN].min(),
        end=data[DATE_COLUMN].max(),
        freq="D",
        name=DATE_COLUMN,
    )
    data = data.set_index(DATE_COLUMN).reindex(full_dates)
    data.index.name = DATE_COLUMN

    # Calendar predictors are known in advance for the forecast date.
    data["day_of_week"] = data.index.dayofweek.astype("int8")
    data["day_of_month"] = data.index.day.astype("int8")
    data["month"] = data.index.month.astype("int8")
    data["day_of_year"] = data.index.dayofyear.astype("int16")
    data["week_of_year"] = data.index.isocalendar().week.astype("int16")
    data["is_weekend"] = (data.index.dayofweek >= 5).astype("int8")
    data["day_of_year_sin"] = np.sin(2 * np.pi * data.index.dayofyear / 365.25)
    data["day_of_year_cos"] = np.cos(2 * np.pi * data.index.dayofyear / 365.25)

    # All demand-derived features are shifted so row D never uses demand on D.
    for target in TARGET_COLUMNS:
        past = data[target]

        for lag in LAG_DAYS:
            data[f"{target}_lag_{lag}d"] = past.shift(lag)

        for window in ROLLING_WINDOWS:
            shifted = past.shift(1)
            data[f"{target}_rolling_mean_{window}d"] = (
                shifted.rolling(window=window, min_periods=window).mean()
            )
            data[f"{target}_rolling_std_{window}d"] = (
                shifted.rolling(window=window, min_periods=window).std()
            )
            data[f"{target}_rolling_min_{window}d"] = (
                shifted.rolling(window=window, min_periods=window).min()
            )
            data[f"{target}_rolling_max_{window}d"] = (
                shifted.rolling(window=window, min_periods=window).max()
            )

        # Change between the two most recent fully observed days before D.
        data[f"{target}_change_1d_pct"] = (
            past.shift(1).pct_change(periods=1, fill_method=None) * 100.0
        )
        data[f"{target}_change_7d_pct"] = (
            past.shift(1).pct_change(periods=7, fill_method=None) * 100.0
        )

    result = data.reset_index()
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def build_demand_features() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Electricity input not found: {INPUT_PATH}\n"
            "Run `python -m ml.data.electricity_loader` first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(INPUT_PATH)
    features = engineer_demand_features(source)
    features.to_csv(OUTPUT_PATH, index=False)

    missing_calendar_rows = int(features[TARGET_COLUMNS].isna().all(axis=1).sum())
    report = {
        "geography": "West Bengal",
        "frequency": "daily",
        "input_file": str(INPUT_PATH.relative_to(ROOT)),
        "output_file": str(OUTPUT_PATH.relative_to(ROOT)),
        "rows_on_complete_calendar": int(len(features)),
        "date_start": features[DATE_COLUMN].min().strftime("%Y-%m-%d"),
        "date_end": features[DATE_COLUMN].max().strftime("%Y-%m-%d"),
        "source_rows": int(len(source)),
        "calendar_rows_without_both_targets": missing_calendar_rows,
        "target_columns": TARGET_COLUMNS,
        "lag_days": list(LAG_DAYS),
        "rolling_windows_days": list(ROLLING_WINDOWS),
        "causality_policy": (
            "Every demand lag, rolling statistic and percentage change for date D "
            "uses demand observations strictly before D. Missing calendar dates "
            "remain missing and are not interpolated."
        ),
        "feature_columns": [
            column for column in features.columns
            if column not in [DATE_COLUMN, *TARGET_COLUMNS]
        ],
        "missing_by_column": {
            column: int(count)
            for column, count in features.isna().sum().items()
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Demand feature engineering completed.")
    logger.info("Rows: %s", len(features))
    logger.info("Calendar rows with missing targets: %s", missing_calendar_rows)
    logger.info("Features created: %s", len(report["feature_columns"]))
    logger.info("Saved dataset: %s", OUTPUT_PATH)
    logger.info("Saved report: %s", REPORT_PATH)
    return features


if __name__ == "__main__":
    build_demand_features()
