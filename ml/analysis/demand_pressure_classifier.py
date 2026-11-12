"""
GridGuard AI - Historical Demand Pressure Classifier

Purpose:
1. Learn statistical demand-pressure thresholds from the demand-change
   model's training period only.
2. Apply those fixed thresholds to the complete historical dataset.
3. Export labelled history and a JSON threshold/report file.

Important:
- These are demand-pressure categories, NOT verified grid-overload labels.
- No generation capacity or reserve-margin data is available.
- Thresholds are learned from training-period actual demand only.
- Do not recompute thresholds using validation/test/future data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "west_bengal_demand_weather_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_pressure"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "demand_pressure"
)

# Demand-change training split: 2 Jan 2013 through 13 Feb 2021.
# This cutoff is deliberately fixed to prevent threshold leakage.
TRAIN_END_DATE = "2021-02-13"

TARGETS = {
    "energy_MU": "energy",
    "peak_demand_MW": "peak_demand",
}

PERCENTILES = {
    "elevated": 80,
    "high": 90,
    "critical": 95,
}

# Category order is used when combining energy and peak pressure.
CATEGORY_ORDER = {
    "normal": 0,
    "elevated": 1,
    "high": 2,
    "critical": 3,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("demand_pressure_classifier")


def calculate_thresholds(
    train_df: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Calculate P80/P90/P95 from training-period actual targets only."""
    thresholds: dict[str, dict[str, float]] = {}

    for target, label in TARGETS.items():
        values = pd.to_numeric(
            train_df[target], errors="coerce"
        ).dropna()

        if len(values) < 100:
            raise ValueError(
                f"Insufficient training observations for {target}: {len(values)}"
            )

        thresholds[label] = {
            "p80": float(values.quantile(0.80)),
            "p90": float(values.quantile(0.90)),
            "p95": float(values.quantile(0.95)),
            "training_min": float(values.min()),
            "training_max": float(values.max()),
            "training_mean": float(values.mean()),
            "training_observations": int(values.shape[0]),
        }

    return thresholds


def classify_value(
    value: Any,
    threshold: dict[str, float],
) -> str:
    """Classify a single demand value using fixed training thresholds."""
    if pd.isna(value):
        return "unknown"

    value = float(value)

    if value >= threshold["p95"]:
        return "critical"
    if value >= threshold["p90"]:
        return "high"
    if value >= threshold["p80"]:
        return "elevated"
    return "normal"


def apply_pressure_labels(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Apply demand-pressure labels without changing original target values."""
    result = df.copy()

    result["energy_pressure"] = result["energy_MU"].apply(
        lambda x: classify_value(x, thresholds["energy"])
    )

    result["peak_pressure"] = result["peak_demand_MW"].apply(
        lambda x: classify_value(x, thresholds["peak_demand"])
    )

    def combined_category(row: pd.Series) -> str:
        categories = [
            row["energy_pressure"],
            row["peak_pressure"],
        ]
        known = [
            category
            for category in categories
            if category in CATEGORY_ORDER
        ]

        if not known:
            return "unknown"

        return max(known, key=lambda item: CATEGORY_ORDER[item])

    result["demand_pressure"] = result.apply(
        combined_category, axis=1
    )

    result["demand_pressure_score"] = result[
        "demand_pressure"
    ].map(CATEGORY_ORDER)

    result["is_elevated_or_above"] = result[
        "demand_pressure_score"
    ].ge(CATEGORY_ORDER["elevated"])

    result["is_high_or_critical"] = result[
        "demand_pressure_score"
    ].ge(CATEGORY_ORDER["high"])

    return result


def build_report(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
    labelled: pd.DataFrame,
) -> dict[str, Any]:
    """Build a transparent report describing threshold construction."""
    category_counts = (
        labelled["demand_pressure"]
        .value_counts(dropna=False)
        .to_dict()
    )

    return {
        "project": "GridGuard AI",
        "module": "historical_demand_pressure_classifier",
        "interpretation": (
            "Statistical demand-pressure classification only. "
            "It does not establish physical grid overload or insufficient reserve."
        ),
        "threshold_method": "training_period_actual_demand_percentiles",
        "training_end_date": TRAIN_END_DATE,
        "percentile_definitions": PERCENTILES,
        "thresholds": thresholds,
        "dataset": {
            "input_path": str(INPUT_PATH),
            "total_rows": int(len(df)),
            "training_rows_with_valid_targets": int(len(train_df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
        },
        "label_counts": {
            str(key): int(value)
            for key, value in category_counts.items()
        },
        "limitations": [
            "No official available-capacity or reserve-margin data was supplied.",
            "Labels represent relative demand pressure, not verified grid stress.",
            "Energy and peak thresholds are based only on the training period.",
            "Weather variables are not used in this first-stage classifier.",
            "Unknown target values remain unclassified.",
        ],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    logger.info("Loading fused demand-weather dataset: %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    required = {"date", "energy_MU", "peak_demand_MW"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("date").reset_index(drop=True)

    train_end = pd.Timestamp(TRAIN_END_DATE)

    train_df = df.loc[
        (df["date"] <= train_end)
        & df["energy_MU"].notna()
        & df["peak_demand_MW"].notna()
    ].copy()

    if train_df.empty:
        raise ValueError(
            "No valid training-period observations found. "
            "Check TRAIN_END_DATE and dataset dates."
        )

    logger.info(
        "Training threshold reference: %s to %s (%d rows)",
        train_df["date"].min().date(),
        train_df["date"].max().date(),
        len(train_df),
    )

    thresholds = calculate_thresholds(train_df)
    labelled = apply_pressure_labels(df, thresholds)

    output_csv = OUTPUT_DIR / "historical_demand_pressure.csv"
    threshold_json = REPORT_DIR / "demand_pressure_thresholds.json"
    report_json = REPORT_DIR / "demand_pressure_report.json"

    labelled.to_csv(output_csv, index=False)

    with threshold_json.open("w", encoding="utf-8") as file:
        json.dump(thresholds, file, indent=4)

    report = build_report(df, train_df, thresholds, labelled)
    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info("Thresholds:")
    for target, values in thresholds.items():
        logger.info(
            "%s | P80=%.3f | P90=%.3f | P95=%.3f",
            target,
            values["p80"],
            values["p90"],
            values["p95"],
        )

    logger.info(
        "Demand pressure distribution:\n%s",
        labelled["demand_pressure"].value_counts(dropna=False).to_string(),
    )
    logger.info("Saved labelled data: %s", output_csv)
    logger.info("Saved thresholds: %s", threshold_json)
    logger.info("Saved report: %s", report_json)
    logger.info("Demand-pressure classification completed.")


if __name__ == "__main__":
    main()
