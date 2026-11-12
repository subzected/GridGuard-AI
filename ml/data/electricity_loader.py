"""Load and validate West Bengal daily electricity demand datasets.

Expected input files:
    data/raw/electricity/corrected_daily_energy_met_MU.csv
    data/raw/electricity/corrected_peak_met_MW.csv

Outputs:
    data/processed/electricity_daily_cleaned.csv
    data/processed/electricity_energy_cleaned.csv
    data/processed/electricity_peak_cleaned.csv
    reports/electricity/electricity_data_quality_report.json

The loader deliberately drops dates with conflicting West Bengal target values.
It does not average or arbitrarily select conflicting source records.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "electricity"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "electricity"

ENERGY_FILE = "corrected_daily_energy_met_MU.csv"
PEAK_FILE = "corrected_peak_met_MW.csv"
DATE_COLUMN = "Date"
STATE_COLUMN = "West Bengal"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_state_series(path: Path, target_name: str, output_name: str) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Place both downloaded electricity CSV files in data/raw/electricity/."
        )

    df = pd.read_csv(path)
    required = {DATE_COLUMN, STATE_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {sorted(missing)}")

    original_rows = len(df)
    df = df[[DATE_COLUMN, STATE_COLUMN]].copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce", dayfirst=False)
    df[STATE_COLUMN] = pd.to_numeric(df[STATE_COLUMN], errors="coerce")

    invalid_dates = int(df[DATE_COLUMN].isna().sum())
    invalid_target = int(df[STATE_COLUMN].isna().sum())
    df = df.dropna(subset=[DATE_COLUMN, STATE_COLUMN])
    df[DATE_COLUMN] = df[DATE_COLUMN].dt.normalize()

    # Remove fully identical rows first.
    before_exact_dedup = len(df)
    df = df.drop_duplicates(subset=[DATE_COLUMN, STATE_COLUMN])
    exact_duplicate_rows_removed = before_exact_dedup - len(df)

    # Do not invent a value when the source gives conflicting values for a date.
    distinct_values = df.groupby(DATE_COLUMN)[STATE_COLUMN].nunique()
    conflicting_dates = distinct_values[distinct_values > 1].index
    conflicting_date_strings = [d.strftime("%Y-%m-%d") for d in conflicting_dates]
    df = df[~df[DATE_COLUMN].isin(conflicting_dates)].copy()

    df = df.rename(columns={DATE_COLUMN: "date", STATE_COLUMN: output_name})
    df = df.sort_values("date").reset_index(drop=True)

    if (df[output_name] < 0).any():
        raise ValueError(f"{path.name}: negative values found in {STATE_COLUMN}.")

    report = {
        "source_file": path.name,
        "target": target_name,
        "original_rows": original_rows,
        "invalid_date_rows_removed": invalid_dates,
        "missing_or_invalid_target_rows_removed": invalid_target,
        "exact_duplicate_rows_removed": exact_duplicate_rows_removed,
        "conflicting_dates_excluded": conflicting_date_strings,
        "clean_rows": len(df),
        "start_date": df["date"].min().strftime("%Y-%m-%d") if not df.empty else None,
        "end_date": df["date"].max().strftime("%Y-%m-%d") if not df.empty else None,
        "missing_target_values": int(df[output_name].isna().sum()),
    }
    return df, report


def build_electricity_dataset() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    energy, energy_report = load_state_series(
        RAW_DIR / ENERGY_FILE, "daily_energy_supplied_MU", "energy_MU"
    )
    peak, peak_report = load_state_series(
        RAW_DIR / PEAK_FILE, "daily_peak_demand_met_MW", "peak_demand_MW"
    )

    energy.to_csv(PROCESSED_DIR / "electricity_energy_cleaned.csv", index=False)
    peak.to_csv(PROCESSED_DIR / "electricity_peak_cleaned.csv", index=False)

    combined = energy.merge(peak, on="date", how="inner", validate="one_to_one")
    combined = combined.sort_values("date").reset_index(drop=True)

    if combined.empty:
        raise ValueError("No overlapping valid dates between energy and peak datasets.")

    # Report calendar gaps; do not fill them with fabricated demand values.
    full_calendar = pd.date_range(combined["date"].min(), combined["date"].max(), freq="D")
    observed = pd.DatetimeIndex(combined["date"])
    missing_dates = full_calendar.difference(observed)

    combined.to_csv(PROCESSED_DIR / "electricity_daily_cleaned.csv", index=False)

    report = {
        "geography": "West Bengal",
        "frequency": "daily",
        "targets": {
            "energy_MU": "Daily energy supplied, MU (million units)",
            "peak_demand_MW": "Daily peak demand met, MW",
        },
        "energy_source_report": energy_report,
        "peak_source_report": peak_report,
        "combined_rows": len(combined),
        "combined_start_date": combined["date"].min().strftime("%Y-%m-%d"),
        "combined_end_date": combined["date"].max().strftime("%Y-%m-%d"),
        "combined_missing_values": {k: int(v) for k, v in combined.isna().sum().items()},
        "missing_calendar_days": len(missing_dates),
        "missing_calendar_date_examples": [d.strftime("%Y-%m-%d") for d in missing_dates[:30]],
        "merge": "inner join on date; only dates present in both cleaned targets retained",
        "duplicate_policy": "Exact duplicates removed; dates with conflicting West Bengal target values excluded, not averaged.",
    }

    report_path = REPORT_DIR / "electricity_data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Electricity dataset built successfully.")
    logger.info("Rows: %s | Date range: %s to %s", len(combined), report["combined_start_date"], report["combined_end_date"])
    logger.info("Missing calendar days: %s", len(missing_dates))
    logger.info("Saved dataset: %s", PROCESSED_DIR / "electricity_daily_cleaned.csv")
    logger.info("Saved report: %s", report_path)
    return combined


if __name__ == "__main__":
    build_electricity_dataset()
