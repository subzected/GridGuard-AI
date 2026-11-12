"""
GridGuard AI - Trend-aware baseline comparison.

Run from project root:
    python -m ml.evaluation.trend_baseline_comparison

Compares daily demand forecasting baselines for:
    - energy_MU
    - peak_demand_MW

Methods:
    1. Persistence: previous calendar day's demand (lag 1)
    2. Seasonal persistence: same weekday from the previous week (lag 7)
    3. 7-day rolling mean
    4. 30-day rolling mean (the available engineered long-window feature)
    5. Trend-adjusted persistence: lag_1 + (lag_1 - lag_7) / 6

The script uses the existing chronological 70/15/15 split.
No model is fitted. Predictions use only precomputed past-demand features.
Each method reports its own valid row count and metrics. A common-row
comparison is also produced so methods can be compared on identical dates.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
REPORT_DIR = ROOT / "reports" / "electricity" / "trend_baselines"
PREDICTION_DIR = ROOT / "data" / "processed" / "trend_baselines"

DATE_COL = "date"
TARGETS = ["energy_MU", "peak_demand_MW"]

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

METHOD_COLUMNS = {
    "persistence_lag_1d": "lag_1",
    "seasonal_persistence_lag_7d": "lag_7",
    "rolling_mean_7d": "rolling_mean_7d",
    "rolling_mean_30d": "rolling_mean_30d",
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])

    required = {DATE_COL, *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if df[DATE_COL].isna().any():
        raise ValueError("Dataset contains invalid dates.")

    if df[DATE_COL].duplicated().any():
        raise ValueError("Duplicate dates found in fused dataset.")

    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    # Keep the full calendar here so chronological boundaries match the
    # prior modelling experiments. Rows without actual targets are ignored
    # only when calculating metrics.
    return df


def chronological_split(df: pd.DataFrame):
    # Match the earlier experiments: split after filtering to valid target rows.
    valid = df.dropna(subset=TARGETS).sort_values(DATE_COL).reset_index(drop=True)

    n = len(valid)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = valid.iloc[:train_end].copy()
    validation = valid.iloc[train_end:validation_end].copy()
    test = valid.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("A chronological split is empty.")

    return train, validation, test


def create_predictions(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    result = pd.DataFrame({
        DATE_COL: frame[DATE_COL].to_numpy(),
        "actual": frame[target].to_numpy(dtype=float),
    })

    lag1_col = f"{target}_lag_1d"
    lag7_col = f"{target}_lag_7d"
    roll7_col = f"{target}_rolling_mean_7d"
    roll30_col = f"{target}_rolling_mean_30d"

    required_feature_columns = [
        lag1_col, lag7_col, roll7_col, roll30_col
    ]
    missing = [c for c in required_feature_columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Missing engineered demand features for {target}: {missing}"
        )

    lag1 = pd.to_numeric(frame[lag1_col], errors="coerce")
    lag7 = pd.to_numeric(frame[lag7_col], errors="coerce")

    result["persistence_lag_1d"] = lag1.to_numpy(dtype=float)
    result["seasonal_persistence_lag_7d"] = lag7.to_numpy(dtype=float)
    result["rolling_mean_7d"] = pd.to_numeric(
        frame[roll7_col], errors="coerce"
    ).to_numpy(dtype=float)
    result["rolling_mean_30d"] = pd.to_numeric(
        frame[roll30_col], errors="coerce"
    ).to_numpy(dtype=float)

    # lag_1 represents D-1 and lag_7 represents D-7.
    # There are six daily intervals between those observations.
    result["trend_adjusted_persistence_7d"] = (
        lag1 + (lag1 - lag7) / 6.0
    ).to_numpy(dtype=float)

    return result


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def evaluate_predictions(predictions: pd.DataFrame, target: str, split_name: str):
    methods = list(METHOD_COLUMNS.keys()) + [
        "trend_adjusted_persistence_7d"
    ]

    results = {}
    valid_masks = {}

    for method in methods:
        values = predictions[method].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(predictions["actual"].to_numpy())
        valid_masks[method] = valid

        if valid.any():
            result = calculate_metrics(
                predictions.loc[valid, "actual"].to_numpy(dtype=float),
                values[valid],
            )
            result["rows_evaluated"] = int(valid.sum())
            results[method] = result
        else:
            results[method] = None

    # Common dates ensure a fair apples-to-apples ranking of all methods.
    common_mask = np.ones(len(predictions), dtype=bool)
    for mask in valid_masks.values():
        common_mask &= mask

    common_results = {}
    if common_mask.any():
        actual = predictions.loc[common_mask, "actual"].to_numpy(dtype=float)
        for method in methods:
            predicted = predictions.loc[common_mask, method].to_numpy(dtype=float)
            common_results[method] = {
                **calculate_metrics(actual, predicted),
                "rows_evaluated": int(common_mask.sum()),
            }

    return {
        "target": target,
        "split": split_name,
        "individual_method_metrics": results,
        "common_row_metrics": common_results,
        "common_rows": int(common_mask.sum()),
    }


def add_split_labels(train, validation, test):
    pieces = []
    for name, frame in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        part = frame[[DATE_COL]].copy()
        part["split"] = name
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def main() -> None:
    setup_logging()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train, validation, test = chronological_split(df)

    logging.info(
        "Valid target observations: %d",
        len(train) + len(validation) + len(test),
    )
    for name, frame in [
        ("TRAIN", train),
        ("VALIDATION", validation),
        ("TEST", test),
    ]:
        logging.info(
            "%s: %d rows | %s to %s",
            name,
            len(frame),
            frame[DATE_COL].min().date(),
            frame[DATE_COL].max().date(),
        )

    split_frames = {
        "train": train,
        "validation": validation,
        "test": test,
    }

    report = {
        "project": "GridGuard AI",
        "dataset": str(DATA_PATH),
        "method": "deterministic trend-aware daily demand baselines",
        "split_method": "chronological 70/15/15 over rows with valid targets",
        "split_dates": {
            name: {
                "rows": int(len(frame)),
                "start": str(frame[DATE_COL].min().date()),
                "end": str(frame[DATE_COL].max().date()),
            }
            for name, frame in split_frames.items()
        },
        "baseline_definitions": {
            "persistence_lag_1d": "Demand on the previous calendar day.",
            "seasonal_persistence_lag_7d": "Demand seven calendar days before the target date.",
            "rolling_mean_7d": "Mean demand over the previous seven calendar days.",
            "rolling_mean_30d": "Mean demand over the previous thirty calendar days; used because this is the available engineered long-window feature.",
            "trend_adjusted_persistence_7d": "lag_1 + (lag_1 - lag_7) / 6; extrapolates the average daily change between D-7 and D-1 by one day.",
        },
        "targets": {},
        "notes": [
            "All predictors are existing prior-demand features and do not use the target date's observed demand.",
            "No imputation is performed; each baseline is evaluated only where its required historical values exist.",
            "Individual-method metrics can use different row counts.",
            "Common-row metrics evaluate every baseline on exactly the same dates and should be used for fair ranking.",
            "The 30-day rolling feature is used instead of a 28-day feature because the current feature dataset contains rolling_mean_30d.",
            "These are daily state-level demand baselines, not feeder-level forecasts.",
        ],
    }

    prediction_outputs = []

    for target in TARGETS:
        report["targets"][target] = {}
        for split_name, frame in split_frames.items():
            predictions = create_predictions(frame, target)
            evaluation = evaluate_predictions(predictions, target, split_name)
            report["targets"][target][split_name] = evaluation

            predictions.insert(1, "target", target)
            predictions.insert(2, "split", split_name)
            prediction_outputs.append(predictions)

            if split_name in ("validation", "test"):
                common = evaluation["common_row_metrics"]
                logging.info("%s | %s | common rows: %d",
                             target, split_name, evaluation["common_rows"])
                for method, metrics in common.items():
                    logging.info(
                        "%s | %s | %s | MAE=%.3f | RMSE=%.3f | R2=%.4f",
                        target, split_name, method,
                        metrics["MAE"], metrics["RMSE"], metrics["R2"],
                    )

    report_path = REPORT_DIR / "trend_baseline_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    predictions_path = PREDICTION_DIR / "trend_baseline_predictions.csv"
    pd.concat(prediction_outputs, ignore_index=True).to_csv(
        predictions_path, index=False
    )

    summary_rows = []
    for target, target_report in report["targets"].items():
        for split_name in ("validation", "test"):
            evaluation = target_report[split_name]
            for method, values in evaluation["common_row_metrics"].items():
                summary_rows.append({
                    "target": target,
                    "split": split_name,
                    "method": method,
                    "rows_evaluated": values["rows_evaluated"],
                    "MAE": values["MAE"],
                    "RMSE": values["RMSE"],
                    "R2": values["R2"],
                })

    summary_path = REPORT_DIR / "trend_baseline_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    logging.info("Trend baseline comparison completed.")
    logging.info("JSON report: %s", report_path)
    logging.info("Summary CSV: %s", summary_path)
    logging.info("Predictions CSV: %s", predictions_path)


if __name__ == "__main__":
    main()
