"""
Diagnostics for the saved LightGBM demand models.

Loads the already-trained models and their saved test predictions.
Does NOT train, refit, or overwrite any model.

Run from project root:
    python -m ml.evaluation.advanced_model_diagnostics
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
PREDICTIONS_PATH = (
    ROOT / "data" / "processed" / "advanced_demand"
    / "advanced_demand_test_predictions.csv"
)
MODEL_DIR = ROOT / "ml" / "models" / "advanced_demand"
OUTPUT_DIR = ROOT / "reports" / "electricity" / "advanced_diagnostics"
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

TARGETS = {
    "energy_MU": {
        "actual": "energy_MU_actual",
        "predicted": "energy_MU_predicted",
        "unit": "MU",
    },
    "peak_demand_MW": {
        "actual": "peak_demand_MW_actual",
        "predicted": "peak_demand_MW_predicted",
        "unit": "MW",
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("advanced_model_diagnostics")


def metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    residual = actual_values - predicted_values

    return {
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
        "mean_error_actual_minus_predicted": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "max_absolute_error": float(np.max(np.abs(residual))),
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path in (DATASET_PATH, PREDICTIONS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Check that the dataset and advanced training outputs exist."
            )

    dataset = pd.read_csv(DATASET_PATH, parse_dates=["date"])
    predictions = pd.read_csv(PREDICTIONS_PATH, parse_dates=["date"])

    if dataset["date"].duplicated().any():
        raise ValueError("The fused dataset contains duplicate dates.")
    if predictions["date"].duplicated().any():
        raise ValueError("The test predictions contain duplicate dates.")

    merged = predictions.merge(
        dataset,
        on="date",
        how="left",
        validate="one_to_one",
        suffixes=("", "_dataset"),
    )

    for target, columns in TARGETS.items():
        if columns["actual"] not in merged or columns["predicted"] not in merged:
            raise ValueError(
                f"Prediction file is missing columns for {target}: "
                f"{columns['actual']}, {columns['predicted']}"
            )
        if merged[columns["actual"]].isna().any():
            raise ValueError(f"Actual values are missing for {target} in predictions.")
        if merged[columns["predicted"]].isna().any():
            raise ValueError(f"Predictions are missing for {target}.")

    if merged["date"].isna().any():
        raise ValueError("Some prediction dates could not be matched to the dataset.")

    return dataset, merged.sort_values("date").reset_index(drop=True)


def load_model_bundle(target: str) -> dict[str, Any]:
    path = MODEL_DIR / f"{target}_lightgbm.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Saved LightGBM model not found: {path}\n"
            "Copy the selected model into ml/models/advanced_demand/."
        )

    # Only load trusted model files created by this project.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(path)
    for warning in caught:
        logger.warning("Model loading warning for %s: %s", target, warning.message)

    required = {"pipeline", "target", "feature_columns", "model_name"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle for {target} is missing: {sorted(missing)}")
    if bundle["target"] != target:
        raise ValueError(
            f"Model target mismatch: expected {target}, got {bundle['target']}"
        )
    return bundle


def feature_importance_table(bundle: dict[str, Any]) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    model = pipeline.named_steps["model"]
    features = list(bundle["feature_columns"])

    if hasattr(model, "booster_"):
        gain = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
    else:
        gain = getattr(model, "feature_importances_", np.zeros(len(features)))
        split = np.zeros(len(features))

    table = pd.DataFrame({
        "feature": features,
        "gain_importance": np.asarray(gain, dtype=float),
        "split_importance": np.asarray(split, dtype=float),
    })
    total_gain = table["gain_importance"].sum()
    total_split = table["split_importance"].sum()
    table["gain_importance_pct"] = (
        table["gain_importance"] / total_gain * 100 if total_gain else 0.0
    )
    table["split_importance_pct"] = (
        table["split_importance"] / total_split * 100 if total_split else 0.0
    )
    return table.sort_values("gain_importance", ascending=False).reset_index(drop=True)


def add_feature_group(feature: str) -> str:
    if feature.startswith(("imd_", "open_meteo_")):
        return "Weather"
    if "_lag_" in feature or "_rolling_" in feature or "_change_" in feature:
        return "Demand history"
    if feature in {
        "day_of_week", "day_of_month", "month", "day_of_year",
        "week_of_year", "is_weekend", "day_of_year_sin", "day_of_year_cos",
    }:
        return "Calendar"
    return "Other"


def plot_actual_vs_predicted(frame: pd.DataFrame, target: str, unit: str) -> None:
    actual_col = TARGETS[target]["actual"]
    pred_col = TARGETS[target]["predicted"]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(frame["date"], frame[actual_col], label="Actual", linewidth=1.3)
    ax.plot(frame["date"], frame[pred_col], label="LightGBM prediction", linewidth=1.1)
    ax.set_title(f"{target}: actual vs predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel(unit)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def plot_residuals(frame: pd.DataFrame, target: str, unit: str) -> None:
    actual_col = TARGETS[target]["actual"]
    pred_col = TARGETS[target]["predicted"]
    residual = frame[actual_col] - frame[pred_col]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.plot(frame["date"], residual, linewidth=1)
    ax.set_title(f"{target}: residuals over time (actual - predicted)")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Residual ({unit})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_residuals_over_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(residual, bins=35, edgecolor="black", alpha=0.8)
    ax.axvline(0, linewidth=1, linestyle="--")
    ax.set_title(f"{target}: residual distribution")
    ax.set_xlabel(f"Residual ({unit})")
    ax.set_ylabel("Number of days")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_residual_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(frame[actual_col], residual, alpha=0.55, s=16)
    ax.axhline(0, linewidth=1, linestyle="--")
    ax.set_title(f"{target}: residual vs actual demand")
    ax.set_xlabel(f"Actual ({unit})")
    ax.set_ylabel(f"Residual ({unit})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_residual_vs_actual.png", dpi=160)
    plt.close(fig)


def plot_feature_importance(table: pd.DataFrame, target: str) -> None:
    top = table.head(20).sort_values("gain_importance_pct")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top["feature"], top["gain_importance_pct"])
    ax.set_title(f"{target}: top 20 LightGBM features (gain importance)")
    ax.set_xlabel("Share of total gain importance (%)")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_feature_importance_top20.png", dpi=160)
    plt.close(fig)


def build_monthly_errors(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    actual_col = TARGETS[target]["actual"]
    pred_col = TARGETS[target]["predicted"]
    work = frame[["date", actual_col, pred_col]].copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    work["absolute_error"] = (work[actual_col] - work[pred_col]).abs()
    work["signed_error"] = work[actual_col] - work[pred_col]

    monthly = work.groupby("month", as_index=False).agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("signed_error", "mean"),
        actual_mean=(actual_col, "mean"),
        predicted_mean=(pred_col, "mean"),
    )
    return monthly


def build_demand_range_errors(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    actual_col = TARGETS[target]["actual"]
    pred_col = TARGETS[target]["predicted"]
    work = frame[[actual_col, pred_col]].copy()
    work["absolute_error"] = (work[actual_col] - work[pred_col]).abs()
    work["signed_error"] = work[actual_col] - work[pred_col]

    try:
        work["actual_demand_range"] = pd.qcut(
            work[actual_col], q=5, duplicates="drop"
        ).astype(str)
    except ValueError:
        work["actual_demand_range"] = "all"

    return work.groupby("actual_demand_range", as_index=False, observed=False).agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_signed_error=("signed_error", "mean"),
        actual_mean=(actual_col, "mean"),
    )


def plot_monthly_errors(monthly: pd.DataFrame, target: str, unit: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.bar(monthly["month"], monthly["mae"])
    ax.set_title(f"{target}: monthly test MAE")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"MAE ({unit})")
    ax.tick_params(axis="x", rotation=75)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / f"{target}_monthly_mae.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    dataset, test_frame = load_inputs()
    logger.info(
        "Loaded dataset: %d rows; test predictions: %d rows",
        len(dataset), len(test_frame),
    )
    logger.info(
        "Test period: %s to %s",
        test_frame["date"].min().date(), test_frame["date"].max().date(),
    )

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Post-training diagnostics only; no model fitting or selection.",
        "test_period": {
            "start": str(test_frame["date"].min().date()),
            "end": str(test_frame["date"].max().date()),
            "rows": int(len(test_frame)),
        },
        "targets": {},
        "outputs": {},
    }

    combined_importance = []
    combined_monthly = []
    combined_ranges = []
    residual_output = test_frame[["date"]].copy()

    for target, config in TARGETS.items():
        logger.info("Diagnostics for %s", target)
        actual_col = config["actual"]
        pred_col = config["predicted"]
        unit = config["unit"]

        frame = test_frame[["date", actual_col, pred_col]].copy()
        frame["residual"] = frame[actual_col] - frame[pred_col]
        frame["absolute_error"] = frame["residual"].abs()

        bundle = load_model_bundle(target)
        importance = feature_importance_table(bundle)
        importance["target"] = target
        importance["feature_group"] = importance["feature"].map(add_feature_group)
        importance.to_csv(TABLE_DIR / f"{target}_feature_importance.csv", index=False)
        combined_importance.append(importance)

        plot_actual_vs_predicted(test_frame, target, unit)
        plot_residuals(test_frame, target, unit)
        plot_feature_importance(importance, target)

        monthly = build_monthly_errors(test_frame, target)
        monthly["target"] = target
        monthly.to_csv(TABLE_DIR / f"{target}_monthly_errors.csv", index=False)
        combined_monthly.append(monthly)
        plot_monthly_errors(monthly, target, unit)

        demand_ranges = build_demand_range_errors(test_frame, target)
        demand_ranges["target"] = target
        demand_ranges.to_csv(
            TABLE_DIR / f"{target}_demand_range_errors.csv", index=False
        )
        combined_ranges.append(demand_ranges)

        # The training module's persistence feature is named *_lag_1d,
        # not *_lag_1. Evaluate on the exact same test rows where it exists.
        lag_col = f"{target}_lag_1d"
        baseline_result = None
        model_on_baseline_rows = None
        if lag_col in test_frame.columns:
            baseline_mask = test_frame[lag_col].notna()
            if baseline_mask.any():
                baseline_result = metrics(
                    test_frame.loc[baseline_mask, actual_col],
                    test_frame.loc[baseline_mask, lag_col],
                )
                baseline_result["rows"] = int(baseline_mask.sum())
                # Fair comparison: evaluate LightGBM on precisely the same
                # dates for which the persistence prediction is available.
                model_on_baseline_rows = metrics(
                    test_frame.loc[baseline_mask, actual_col],
                    test_frame.loc[baseline_mask, pred_col],
                )
                model_on_baseline_rows["rows"] = int(baseline_mask.sum())
        else:
            logger.warning("Persistence feature %s not found.", lag_col)

        test_metrics = metrics(frame[actual_col], frame[pred_col])
        importance_by_group = (
            importance.groupby("feature_group")["gain_importance_pct"]
            .sum().sort_values(ascending=False).to_dict()
        )

        top_features = importance.head(15)[
            ["feature", "gain_importance_pct", "split_importance"]
        ].to_dict(orient="records")

        worst_days = frame.nlargest(15, "absolute_error")[
            ["date", actual_col, pred_col, "residual", "absolute_error"]
        ].copy()
        worst_days["date"] = worst_days["date"].dt.strftime("%Y-%m-%d")
        worst_days.to_csv(
            TABLE_DIR / f"{target}_worst_15_days.csv", index=False
        )

        report["targets"][target] = {
            "model_name": bundle["model_name"],
            "test_metrics": test_metrics,
            "model_metrics_on_persistence_comparable_rows": model_on_baseline_rows,
            "persistence_baseline_metrics": baseline_result,
            "feature_group_gain_importance_pct": {
                key: float(value) for key, value in importance_by_group.items()
            },
            "top_15_features": top_features,
            "monthly_mae_max": (
                monthly.loc[monthly["mae"].idxmax()].to_dict()
                if not monthly.empty else None
            ),
            "monthly_mae_min": (
                monthly.loc[monthly["mae"].idxmin()].to_dict()
                if not monthly.empty else None
            ),
            "demand_range_errors": demand_ranges.to_dict(orient="records"),
            "worst_15_test_days": worst_days.to_dict(orient="records"),
        }

        residual_output[f"{target}_residual"] = frame["residual"].to_numpy()
        residual_output[f"{target}_absolute_error"] = frame["absolute_error"].to_numpy()

        logger.info(
            "%s | model MAE=%.4f RMSE=%.4f R2=%.4f",
            target, test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )
        if baseline_result and model_on_baseline_rows:
            logger.info(
                "%s | comparable rows=%d | LightGBM MAE=%.4f | persistence MAE=%.4f",
                target, baseline_result["rows"],
                model_on_baseline_rows["mae"], baseline_result["mae"],
            )

    importance_all = pd.concat(combined_importance, ignore_index=True)
    monthly_all = pd.concat(combined_monthly, ignore_index=True)
    ranges_all = pd.concat(combined_ranges, ignore_index=True)

    importance_all.to_csv(TABLE_DIR / "feature_importance_all_targets.csv", index=False)
    monthly_all.to_csv(TABLE_DIR / "monthly_errors_all_targets.csv", index=False)
    ranges_all.to_csv(TABLE_DIR / "demand_range_errors_all_targets.csv", index=False)
    residual_path = TABLE_DIR / "test_residuals.csv"
    residual_output.to_csv(residual_path, index=False)

    report_path = OUTPUT_DIR / "advanced_model_diagnostics_report.json"
    report["outputs"] = {
        "plots_directory": str(PLOT_DIR),
        "tables_directory": str(TABLE_DIR),
        "residuals_csv": str(residual_path),
        "feature_importance_csv": str(TABLE_DIR / "feature_importance_all_targets.csv"),
        "monthly_errors_csv": str(TABLE_DIR / "monthly_errors_all_targets.csv"),
        "report_json": str(report_path),
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    logger.info("Diagnostics completed successfully.")
    logger.info("Report: %s", report_path)
    logger.info("Plots: %s", PLOT_DIR)
    logger.info("Tables: %s", TABLE_DIR)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Advanced model diagnostics failed.")
        sys.exit(1)
