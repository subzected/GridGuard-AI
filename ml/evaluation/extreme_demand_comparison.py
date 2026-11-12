"""
Head-to-head comparison of GridGuard AI demand forecasting models.

Run from the project root:
    python -m ml.evaluation.extreme_demand_comparison

Compares on the SAME chronological test dates:
1. Persistence (previous-day demand)
2. Trend-adjusted persistence
3. Existing demand-change model bundles
4. Extreme-demand model bundles

The 80th-percentile high-demand threshold is calculated from training actual
demand only. The top 10% test-demand group is reported separately as a
retrospective stress test and is NOT used for model selection.

Outputs:
    reports/electricity/extreme_demand_comparison/
        extreme_demand_comparison_report.json
        tables/
        plots/

This script does not fit or tune models. It only loads saved model bundles
and evaluates them on a common untouched test set.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
DEMAND_CHANGE_DIR = ROOT / "ml" / "models" / "demand_change"
EXTREME_DIR = ROOT / "ml" / "models" / "extreme_demand"
OUTPUT_DIR = ROOT / "reports" / "electricity" / "extreme_demand_comparison"
TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

TARGETS = {
    "energy_MU": {"unit": "MU", "label": "Daily energy"},
    "peak_demand_MW": {"unit": "MW", "label": "Peak demand"},
}
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
HIGH_DEMAND_QUANTILE = 0.80
TOP_TEST_QUANTILE = 0.90

LOG = logging.getLogger("extreme_demand_comparison")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def make_directories() -> None:
    for path in (OUTPUT_DIR, TABLE_DIR, PLOT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    for target in TARGETS:
        if target not in df.columns:
            raise ValueError(f"Dataset is missing target: {target}")

    return df


def get_lag_column(df: pd.DataFrame, target: str, lag: int) -> str:
    candidates = (
        f"{target}_lag_{lag}d",
        f"{target}_lag{lag}",
        f"{target}_lag_{lag}",
    )
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find {lag}-day lag column for {target}. "
        f"Tried: {candidates}"
    )


def get_common_frame(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Preserve all model features and match the training modules' row eligibility.

    Both targets and both 1-day lags determine eligible rows. The 7-day lags
    are retained for trend-adjusted persistence but do not determine the
    chronological split, matching the model-training split.
    """
    lag1_columns = {
        target: get_lag_column(df, target, 1)
        for target in TARGETS
    }

    lag7_columns = {
        target: get_lag_column(df, target, 7)
        for target in TARGETS
    }

    # Keep the complete original dataframe because the saved model bundles
    # require calendar, demand-history, and weather feature columns.
    frame = df.copy()

    frame = (
        frame.dropna(
            subset=[
                *TARGETS,
                *lag1_columns.values(),
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(frame) < 100:
        raise ValueError(
            f"Only {len(frame)} usable rows found."
        )

    return frame, lag1_columns, lag7_columns


def chronological_parts(frame: pd.DataFrame):
    n = len(frame)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    return train, validation, test


def load_bundle(directory: Path, target: str, bundle_type: str) -> tuple[dict, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Model directory not found: {directory}")

    if bundle_type == "demand_change":
        candidates = list(directory.glob(f"{target}_*_delta.joblib"))
    else:
        candidates = list(directory.glob(f"{target}_*_extreme_delta.joblib"))

    if not candidates:
        raise FileNotFoundError(
            f"No {bundle_type} model bundle found for {target} in {directory}"
        )

    # There should normally be one selected bundle per target. If multiple
    # experiment bundles exist, prefer the newest one and record its path.
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise TypeError(f"Expected a dictionary model bundle: {path}")
    if "pipeline" not in bundle:
        raise KeyError(f"'pipeline' missing from model bundle: {path}")

    return bundle, path


def model_predict_level(
    bundle: dict,
    test: pd.DataFrame,
    target: str,
    lag1_col: str,
) -> np.ndarray:
    features = bundle.get("feature_columns") or bundle.get("features")
    if not features:
        raise KeyError(
            f"Feature list missing from {target} model bundle."
        )

    missing = [column for column in features if column not in test.columns]
    if missing:
        raise ValueError(
            f"{target} model requires {len(missing)} missing features: "
            f"{missing[:15]}"
        )

    pipeline = bundle["pipeline"]
    delta = np.asarray(
        pipeline.predict(test[features]),
        dtype=float,
    )
    level = test[lag1_col].to_numpy(dtype=float) + delta

    if len(level) != len(test):
        raise ValueError(
            f"{target} model returned {len(level)} predictions "
            f"for {len(test)} rows."
        )
    return level


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(predicted)
    y = actual[valid]
    p = predicted[valid]

    if len(y) == 0:
        return {
            "n": 0, "mae": None, "rmse": None, "r2": None,
            "mean_error": None, "underprediction_rate_percent": None,
            "overprediction_rate_percent": None, "mape_percent": None,
        }

    errors = p - y
    nonzero = np.abs(y) > 1e-9
    mape = (
        float(np.mean(np.abs(errors[nonzero] / y[nonzero])) * 100)
        if nonzero.any() else None
    )

    return {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(y) > 1 else None,
        "mean_error": float(np.mean(errors)),
        "underprediction_rate_percent": float(np.mean(errors < 0) * 100),
        "overprediction_rate_percent": float(np.mean(errors > 0) * 100),
        "mape_percent": mape,
    }


def build_prediction_frame(
    test: pd.DataFrame,
    target: str,
    lag_columns: dict[str, str],
    lag7_columns: dict[str, str],
    demand_change_bundle: dict,
    extreme_bundle: dict,
) -> pd.DataFrame:
    lag1 = lag_columns[target]
    lag7 = lag7_columns[target]

    result = pd.DataFrame({
        "date": test["date"].to_numpy(),
        "actual": test[target].astype(float).to_numpy(),
        "persistence": test[lag1].astype(float).to_numpy(),
        "trend_adjusted_persistence": (
            test[lag1].astype(float)
            + (test[lag1].astype(float) - test[lag7].astype(float)) / 6.0
        ).to_numpy(),
        "demand_change_model": model_predict_level(
            demand_change_bundle, test, target, lag1
        ),
        "extreme_demand_model": model_predict_level(
            extreme_bundle, test, target, lag1
        ),
    })

    return result


def add_evaluation_masks(
    frame: pd.DataFrame,
    training_threshold: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["is_high_demand_train_p80"] = (
        result["actual"] >= training_threshold
    )
    # Retrospective stress-test group only; never used for training/selection.
    top_test_threshold = float(result["actual"].quantile(TOP_TEST_QUANTILE))
    result["is_top_10pct_test"] = result["actual"] >= top_test_threshold
    result.attrs["top_test_threshold"] = top_test_threshold
    return result


def evaluate_methods(
    frame: pd.DataFrame,
    methods: list[str],
) -> dict[str, Any]:
    actual = frame["actual"].to_numpy(dtype=float)
    high_mask = frame["is_high_demand_train_p80"].to_numpy(dtype=bool)
    top_mask = frame["is_top_10pct_test"].to_numpy(dtype=bool)

    output = {}
    for method in methods:
        prediction = frame[method].to_numpy(dtype=float)
        output[method] = {
            "overall": calculate_metrics(actual, prediction),
            "high_demand_train_p80": calculate_metrics(
                actual[high_mask], prediction[high_mask]
            ),
            "top_10_percent_test_stress_group": calculate_metrics(
                actual[top_mask], prediction[top_mask]
            ),
        }

    return output


def build_monthly_table(
    frame: pd.DataFrame,
    methods: list[str],
    target: str,
) -> pd.DataFrame:
    rows = []
    work = frame.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)

    for month, group in work.groupby("month", sort=True):
        high = group[group["is_high_demand_train_p80"]]
        for method in methods:
            overall_metrics = calculate_metrics(
                group["actual"].to_numpy(),
                group[method].to_numpy(),
            )
            high_metrics = calculate_metrics(
                high["actual"].to_numpy(),
                high[method].to_numpy(),
            )
            rows.append({
                "target": target,
                "month": month,
                "method": method,
                "overall_n": overall_metrics["n"],
                "overall_mae": overall_metrics["mae"],
                "overall_rmse": overall_metrics["rmse"],
                "high_demand_n": high_metrics["n"],
                "high_demand_mae": high_metrics["mae"],
                "high_demand_rmse": high_metrics["rmse"],
                "high_demand_mean_error": high_metrics["mean_error"],
                "high_demand_underprediction_rate_percent": (
                    high_metrics["underprediction_rate_percent"]
                ),
            })

    return pd.DataFrame(rows)


def build_error_table(
    frame: pd.DataFrame,
    methods: list[str],
    target: str,
) -> pd.DataFrame:
    result = frame[[
        "date", "actual",
        "is_high_demand_train_p80",
        "is_top_10pct_test",
    ]].copy()
    result.insert(0, "target", target)

    for method in methods:
        residual = frame[method].astype(float) - frame["actual"].astype(float)
        result[f"{method}_prediction"] = frame[method].astype(float)
        result[f"{method}_error"] = residual
        result[f"{method}_absolute_error"] = residual.abs()
        result[f"{method}_underpredicted"] = residual < 0

    return result


def plot_overall_comparison(
    metrics: dict[str, Any], target: str
) -> None:
    methods = list(metrics.keys())
    overall = [metrics[m]["overall"]["mae"] for m in methods]
    high = [metrics[m]["high_demand_train_p80"]["mae"] for m in methods]

    x = np.arange(len(methods))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, overall, width, label="All test days")
    ax.bar(x + width / 2, high, width, label="Train P80 high-demand days")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right")
    ax.set_ylabel(f"MAE ({TARGETS[target]['unit']})")
    ax.set_title(f"{target}: model comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_mae_comparison.png")


def plot_high_demand_underprediction(
    metrics: dict[str, Any], target: str
) -> None:
    methods = list(metrics.keys())
    rates = [
        metrics[m]["high_demand_train_p80"]["underprediction_rate_percent"]
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(methods, rates)
    ax.set_ylabel("Underprediction rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{target}: high-demand underprediction frequency")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_high_demand_underprediction.png")


def plot_predictions(
    frame: pd.DataFrame,
    methods: list[str],
    target: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["date"], frame["actual"], label="Actual", linewidth=1.8)
    for method in methods:
        ax.plot(frame["date"], frame[method], label=method, alpha=0.8)
    ax.set_title(f"{target}: actual vs model predictions")
    ax.set_ylabel(TARGETS[target]["unit"])
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.legend()
    ax.grid(alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_actual_vs_predictions.png")


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def main() -> None:
    setup_logging()
    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*",
    )
    make_directories()

    dataset = load_dataset()
    frame, lag_columns, lag7_columns = get_common_frame(dataset)
    train, validation, test = chronological_parts(frame)

    LOG.info("Loaded dataset: %s", DATA_PATH)
    LOG.info(
        "Common split rows: train=%d validation=%d test=%d",
        len(train), len(validation), len(test),
    )
    LOG.info(
        "Test dates: %s to %s",
        test["date"].min().date(),
        test["date"].max().date(),
    )

    all_predictions = []
    report: dict[str, Any] = {
        "dataset": str(DATA_PATH),
        "comparison_policy": (
            "All models are re-evaluated on the same chronological test rows. "
            "No model is fitted or selected in this script."
        ),
        "split": {
            "train": {
                "rows": len(train),
                "start": train["date"].min().date().isoformat(),
                "end": train["date"].max().date().isoformat(),
            },
            "validation": {
                "rows": len(validation),
                "start": validation["date"].min().date().isoformat(),
                "end": validation["date"].max().date().isoformat(),
            },
            "test": {
                "rows": len(test),
                "start": test["date"].min().date().isoformat(),
                "end": test["date"].max().date().isoformat(),
            },
        },
        "high_demand_quantile": HIGH_DEMAND_QUANTILE,
        "top_test_quantile": TOP_TEST_QUANTILE,
        "targets": {},
        "caveats": [
            "The high-demand threshold is calculated from training actual demand only.",
            "Actual demand is used retrospectively to form high-demand evaluation groups.",
            "The top 10% test group is a descriptive stress test, not a deployable forecast-time label.",
            "The script does not fit models or tune hyperparameters.",
            "Historical weather features are prior-day observations, not archived operational forecasts.",
            "Latest saved bundle per target is loaded when multiple bundles exist; bundle paths are recorded.",
        ],
    }

    for target in TARGETS:
        LOG.info("========== Comparing %s ==========", target)
        demand_bundle, demand_path = load_bundle(
            DEMAND_CHANGE_DIR, target, "demand_change"
        )
        extreme_bundle, extreme_path = load_bundle(
            EXTREME_DIR, target, "extreme_demand"
        )

        # Training-derived threshold; same split definition as both trainers.
        threshold = float(train[target].astype(float).quantile(HIGH_DEMAND_QUANTILE))

        pred_frame = build_prediction_frame(
            test,
            target,
            lag_columns,
            lag7_columns,
            demand_bundle,
            extreme_bundle,
        )
        pred_frame = add_evaluation_masks(pred_frame, threshold)
        methods = [
            "persistence",
            "trend_adjusted_persistence",
            "demand_change_model",
            "extreme_demand_model",
        ]

        # Enforce a common finite prediction subset across every method.
        numeric_cols = ["actual", *methods]
        finite_mask = np.isfinite(
            pred_frame[numeric_cols].to_numpy(dtype=float)
        ).all(axis=1)
        pred_frame = pred_frame.loc[finite_mask].reset_index(drop=True)

        if pred_frame.empty:
            raise RuntimeError(f"No common finite test predictions for {target}.")

        evaluated = evaluate_methods(pred_frame, methods)
        monthly = build_monthly_table(pred_frame, methods, target)
        errors = build_error_table(pred_frame, methods, target)

        monthly.to_csv(
            TABLE_DIR / f"{target}_monthly_comparison.csv", index=False
        )
        errors.to_csv(
            TABLE_DIR / f"{target}_test_predictions_and_errors.csv",
            index=False,
        )

        high_errors = errors[errors["is_high_demand_train_p80"]].copy()
        high_errors.to_csv(
            TABLE_DIR / f"{target}_high_demand_errors.csv", index=False
        )
        top_errors = errors[errors["is_top_10pct_test"]].copy()
        top_errors.to_csv(
            TABLE_DIR / f"{target}_top_10pct_test_errors.csv", index=False
        )

        plot_overall_comparison(evaluated, target)
        plot_high_demand_underprediction(evaluated, target)
        plot_predictions(pred_frame, methods, target)

        report["targets"][target] = {
            "unit": TARGETS[target]["unit"],
            "test_rows_evaluated": int(len(pred_frame)),
            "test_date_range": {
                "start": pred_frame["date"].min().date().isoformat(),
                "end": pred_frame["date"].max().date().isoformat(),
            },
            "training_p80_threshold": threshold,
            "test_top10_threshold_retrospective": float(
                pred_frame.loc[pred_frame["is_top_10pct_test"], "actual"].min()
            ),
            "model_bundles": {
                "demand_change": str(demand_path),
                "demand_change_selected_model": demand_bundle.get("selected_model"),
                "extreme_demand": str(extreme_path),
                "extreme_demand_selected_model": extreme_bundle.get("selected_model"),
                "demand_change_feature_count": len(
                    demand_bundle.get("feature_columns", [])
                ),
                "extreme_demand_feature_count": len(
                    extreme_bundle.get("feature_columns", [])
                ),
                "feature_lists_identical": (
                    demand_bundle.get("feature_columns")
                    == extreme_bundle.get("feature_columns")
                ),
            },
            "metrics": evaluated,
            "monthly_table": f"tables/{target}_monthly_comparison.csv",
            "high_demand_table": f"tables/{target}_high_demand_errors.csv",
            "top_10pct_table": f"tables/{target}_top_10pct_test_errors.csv",
        }

        LOG.info(
            "%s | n=%d | train P80=%.4f",
            target, len(pred_frame), threshold,
        )
        for method in methods:
            overall = evaluated[method]["overall"]
            high = evaluated[method]["high_demand_train_p80"]
            LOG.info(
                "%s | overall MAE=%.4f | high-demand MAE=%s | "
                "high-demand underprediction=%s%%",
                method,
                overall["mae"],
                (
                    f"{high['mae']:.4f}"
                    if high["mae"] is not None else "n/a"
                ),
                (
                    f"{high['underprediction_rate_percent']:.2f}"
                    if high["underprediction_rate_percent"] is not None
                    else "n/a"
                ),
            )

        all_predictions.append(
            pred_frame.assign(target=target)
        )

    combined = pd.concat(all_predictions, ignore_index=True)
    combined.to_csv(
        TABLE_DIR / "all_targets_comparison_predictions.csv",
        index=False,
    )

    report_path = OUTPUT_DIR / "extreme_demand_comparison_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(report), f, indent=2, default=str)

    LOG.info("Comparison completed.")
    LOG.info("Report: %s", report_path)
    LOG.info("Tables: %s", TABLE_DIR)
    LOG.info("Plots: %s", PLOT_DIR)


if __name__ == "__main__":
    main()
