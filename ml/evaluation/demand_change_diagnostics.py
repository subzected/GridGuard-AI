"""
Diagnostics for demand-change forecasting models.

Run from the project root:
    python -m ml.evaluation.demand_change_diagnostics

Expected input:
    data/processed/demand_change/demand_change_test_predictions.csv

Optional model bundles:
    ml/models/demand_change/*_delta.joblib

Outputs:
    reports/electricity/demand_change_diagnostics/
        demand_change_diagnostics_report.json
        tables/
        plots/

The script deliberately evaluates all methods on the same rows for each target.
It reconstructs demand levels from predicted deltas before calculating errors.
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "demand_change"
    / "demand_change_test_predictions.csv"
)
MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "demand_change"
OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "electricity"
    / "demand_change_diagnostics"
)
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"

TARGETS = {
    "energy_MU": {"unit": "MU", "label": "Daily energy"},
    "peak_demand_MW": {"unit": "MW", "label": "Peak demand"},
}
LOG = logging.getLogger("demand_change_diagnostics")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run python -m ml.training.train_demand_change_models first."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Predictions file is empty: {path}")

    date_col = next(
        (c for c in df.columns if c.lower() in {"date", "ds", "timestamp"}),
        None,
    )
    if date_col is None:
        raise ValueError(
            "Could not identify a date column. "
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").reset_index(drop=True)
    return df


def normalized_column_map(df: pd.DataFrame) -> dict[str, str]:
    """Map simplified column names to original names."""
    result = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]+", "", str(col).lower())
        result[key] = col
    return result


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    required: bool = False,
    description: str = "",
) -> str | None:
    mapping = normalized_column_map(df)
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        if key in mapping:
            return mapping[key]

    # Fallback: tolerate prefixes/suffixes such as target_actual or actual_target.
    keys = list(mapping.items())
    for candidate in candidates:
        key = re.sub(r"[^a-z0-9]+", "", candidate.lower())
        for normalized, original in keys:
            if normalized.endswith(key) or normalized.startswith(key):
                return original

    if required:
        raise ValueError(
            f"Could not identify {description or 'required'} column. "
            f"Tried {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def get_target_columns(
    df: pd.DataFrame, target: str
) -> dict[str, str | None]:
    # The training module's predictions file may use either target-prefixed
    # or role-prefixed column names. Support both naming conventions.
    actual = find_column(
        df,
        [
            f"{target}_actual", f"actual_{target}",
            f"{target}_true", f"true_{target}", target,
        ],
        required=True,
        description=f"{target} actual",
    )
    predicted = find_column(
        df,
        [
            f"{target}_predicted", f"predicted_{target}",
            f"{target}_model_prediction", f"{target}_model_pred",
            f"{target}_prediction", f"{target}_pred",
        ],
        required=True,
        description=f"{target} reconstructed model prediction",
    )
    persistence = find_column(
        df,
        [
            f"{target}_persistence", f"persistence_{target}",
            f"{target}_persistence_prediction",
            f"{target}_lag1_prediction", f"{target}_lag_1_prediction",
        ],
    )
    trend = find_column(
        df,
        [
            f"{target}_trend_adjusted", f"trend_adjusted_{target}",
            f"{target}_trend_prediction",
            f"{target}_trend_adjusted_prediction",
        ],
    )
    lag1 = find_column(
        df,
        [f"{target}_lag_1d", f"{target}_lag1", f"{target}_lag_1"],
    )
    return {
        "actual": actual,
        "model": predicted,
        "persistence": persistence,
        "trend_adjusted": trend,
        "lag1": lag1,
    }


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    pair = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if pair.empty:
        return {
            "n": 0, "mae": None, "rmse": None, "r2": None,
            "mean_error": None, "mean_absolute_percentage_error": None,
        }

    y = pair["actual"].astype(float)
    p = pair["predicted"].astype(float)
    nonzero = y.abs() > 1e-9
    mape = (
        float((np.abs((y[nonzero] - p[nonzero]) / y[nonzero])).mean() * 100)
        if nonzero.any() else None
    )
    return {
        "n": int(len(pair)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(pair) >= 2 else None,
        "mean_error": float((p - y).mean()),
        "mean_absolute_percentage_error": mape,
    }


def evaluate_same_rows(
    df: pd.DataFrame, columns: dict[str, str | None]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    methods = {
        "demand_change_model": columns["model"],
        "persistence": columns["persistence"],
        "trend_adjusted_persistence": columns["trend_adjusted"],
    }
    required_cols = ["date", columns["actual"]] + [
        col for col in methods.values() if col is not None
    ]
    common = df[required_cols].copy()
    common = common.dropna(subset=[columns["actual"]])
    # Use a common evaluation subset for every available comparator.
    available_methods = {k: v for k, v in methods.items() if v is not None}
    if not available_methods:
        raise ValueError("No prediction method columns were found.")

    common = common.dropna(subset=list(available_methods.values()))
    common = common.sort_values("date").reset_index(drop=True)

    actual = common[columns["actual"]].astype(float)
    result = {}
    for method, col in available_methods.items():
        result[method] = calculate_metrics(actual, common[col].astype(float))

    if "persistence" in result and result["persistence"]["mae"]:
        base = result["persistence"]["mae"]
        model_mae = result["demand_change_model"]["mae"]
        result["improvement_vs_persistence_percent"] = (
            float((base - model_mae) / base * 100)
        )
    else:
        result["improvement_vs_persistence_percent"] = None

    if "trend_adjusted_persistence" in result:
        base = result["trend_adjusted_persistence"]["mae"]
        model_mae = result["demand_change_model"]["mae"]
        result["improvement_vs_trend_adjusted_percent"] = (
            float((base - model_mae) / base * 100) if base else None
        )
    else:
        result["improvement_vs_trend_adjusted_percent"] = None

    return common, result


def add_residual_columns(
    frame: pd.DataFrame, actual_col: str, methods: dict[str, str | None]
) -> pd.DataFrame:
    out = frame.copy()
    for method, pred_col in methods.items():
        if pred_col is None:
            continue
        suffix = safe_name(method)
        out[f"{suffix}_residual"] = (
            out[pred_col].astype(float) - out[actual_col].astype(float)
        )
        out[f"{suffix}_absolute_error"] = out[f"{suffix}_residual"].abs()
    return out


def monthly_error_table(
    frame: pd.DataFrame,
    actual_col: str,
    methods: dict[str, str | None],
) -> pd.DataFrame:
    rows = []
    work = frame.copy()
    work["month"] = work["date"].dt.to_period("M").astype(str)
    for month, group in work.groupby("month", sort=True):
        for method, pred_col in methods.items():
            if pred_col is None:
                continue
            metrics = calculate_metrics(
                group[actual_col], group[pred_col]
            )
            rows.append({"month": month, "method": method, **metrics})
    return pd.DataFrame(rows)


def demand_range_table(
    frame: pd.DataFrame,
    actual_col: str,
    methods: dict[str, str | None],
) -> pd.DataFrame:
    work = frame.copy()
    try:
        work["demand_range"] = pd.qcut(
            work[actual_col].astype(float),
            q=5,
            labels=["lowest_20pct", "20_40pct", "40_60pct",
                    "60_80pct", "highest_20pct"],
            duplicates="drop",
        )
    except ValueError:
        work["demand_range"] = "all"

    rows = []
    for demand_range, group in work.groupby("demand_range", observed=True):
        for method, pred_col in methods.items():
            if pred_col is None:
                continue
            rows.append({
                "demand_range": str(demand_range),
                "actual_min": float(group[actual_col].min()),
                "actual_max": float(group[actual_col].max()),
                "method": method,
                **calculate_metrics(group[actual_col], group[pred_col]),
            })
    return pd.DataFrame(rows)


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_actual_vs_predicted(
    frame: pd.DataFrame, actual_col: str, methods: dict[str, str | None],
    target: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(frame["date"], frame[actual_col], label="Actual", linewidth=1.8)
    for method, col in methods.items():
        if col is not None:
            ax.plot(frame["date"], frame[col], label=method, alpha=0.85)
    ax.set_title(f"{target}: actual vs predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel(TARGETS[target]["unit"])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend()
    ax.grid(alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_actual_vs_predicted.png")


def plot_residuals(
    frame: pd.DataFrame, actual_col: str, methods: dict[str, str | None],
    target: str,
) -> None:
    model_col = methods["demand_change_model"]
    residual = frame[model_col].astype(float) - frame[actual_col].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(frame["date"], residual, linewidth=0.9)
    axes[0, 0].axhline(0, linestyle="--")
    axes[0, 0].set_title("Model residuals over time")
    axes[0, 0].set_ylabel(TARGETS[target]["unit"])

    axes[0, 1].hist(residual, bins=35)
    axes[0, 1].axvline(0, linestyle="--")
    axes[0, 1].set_title("Residual distribution")

    axes[1, 0].scatter(frame[actual_col], residual, alpha=0.55, s=16)
    axes[1, 0].axhline(0, linestyle="--")
    axes[1, 0].set_title("Residual vs actual demand")
    axes[1, 0].set_xlabel(f"Actual ({TARGETS[target]['unit']})")

    axes[1, 1].scatter(frame[actual_col], frame[model_col], alpha=0.55, s=16)
    low = min(frame[actual_col].min(), frame[model_col].min())
    high = max(frame[actual_col].max(), frame[model_col].max())
    axes[1, 1].plot([low, high], [low, high], linestyle="--")
    axes[1, 1].set_title("Predicted vs actual")
    axes[1, 1].set_xlabel("Actual")
    axes[1, 1].set_ylabel("Predicted")

    for ax in axes.flat:
        ax.grid(alpha=0.2)
    save_plot(PLOT_DIR / f"{target}_residual_diagnostics.png")


def plot_monthly_errors(monthly: pd.DataFrame, target: str) -> None:
    subset = monthly[monthly["method"].isin(
        ["demand_change_model", "persistence", "trend_adjusted_persistence"]
    )]
    if subset.empty:
        return
    pivot = subset.pivot(index="month", columns="method", values="mae")
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(f"{target}: monthly MAE comparison")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"MAE ({TARGETS[target]['unit']})")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(axis="y", alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_monthly_mae.png")


def load_feature_importance(target: str) -> pd.DataFrame | None:
    candidates = sorted(MODEL_DIR.glob(f"{target}_*_delta.joblib"))
    if not candidates:
        LOG.warning("No model bundle found for %s under %s", target, MODEL_DIR)
        return None

    # The trainer saves one selected bundle per target. Use the most recently
    # modified matching bundle if older experiments are also present.
    path = max(candidates, key=lambda p: p.stat().st_mtime)
    bundle = joblib.load(path)

    pipeline = bundle.get("pipeline") if isinstance(bundle, dict) else None
    if pipeline is None and isinstance(bundle, dict):
        pipeline = bundle.get("model")

    estimator = pipeline
    if hasattr(pipeline, "named_steps"):
        estimator = pipeline.named_steps.get("model", pipeline.steps[-1][1])

    importances = getattr(estimator, "feature_importances_", None)
    feature_names = None
    if isinstance(bundle, dict):
        feature_names = bundle.get("feature_columns") or bundle.get("features")

    if importances is None:
        LOG.info("Selected estimator for %s does not expose feature_importances_", target)
        return None

    values = np.asarray(importances, dtype=float)
    if feature_names is None or len(feature_names) != len(values):
        feature_names = [f"feature_{i}" for i in range(len(values))]

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": values,
    }).sort_values("importance", ascending=False)
    result["importance_percent"] = (
        result["importance"] / result["importance"].sum() * 100
        if result["importance"].sum() > 0 else 0
    )
    return result


def save_feature_importance_plot(
    importance: pd.DataFrame, target: str
) -> None:
    top = importance.head(20).sort_values("importance_percent")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_percent"])
    ax.set_title(f"{target}: top 20 feature importances")
    ax.set_xlabel("Importance (%)")
    ax.grid(axis="x", alpha=0.25)
    save_plot(PLOT_DIR / f"{target}_feature_importance.png")


def main() -> None:
    setup_logging()
    warnings.filterwarnings(
        "ignore", message=".*does not have valid feature names.*"
    )
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_predictions(PREDICTIONS_PATH)
    LOG.info("Loaded predictions: %s | rows=%d", PREDICTIONS_PATH, len(df))
    report: dict[str, Any] = {
        "input_file": str(PREDICTIONS_PATH),
        "evaluation_policy": (
            "All available methods are evaluated on the same non-null rows "
            "for each target. Metrics are computed on reconstructed demand levels."
        ),
        "targets": {},
    }

    all_residuals = []
    for target in TARGETS:
        LOG.info("========== Diagnostics: %s ==========", target)
        columns = get_target_columns(df, target)
        common, metrics = evaluate_same_rows(df, columns)
        methods = {
            "demand_change_model": columns["model"],
            "persistence": columns["persistence"],
            "trend_adjusted_persistence": columns["trend_adjusted"],
        }

        residual_frame = add_residual_columns(
            common, columns["actual"], methods
        )
        residual_frame.to_csv(
            TABLE_DIR / f"{target}_test_residuals.csv", index=False
        )
        all_residuals.append(
            residual_frame.assign(target=target)
        )

        monthly = monthly_error_table(
            common, columns["actual"], methods
        )
        monthly.to_csv(
            TABLE_DIR / f"{target}_monthly_errors.csv", index=False
        )

        ranges = demand_range_table(
            common, columns["actual"], methods
        )
        ranges.to_csv(
            TABLE_DIR / f"{target}_demand_range_errors.csv", index=False
        )

        model_residual_col = "demand_change_model_residual"
        worst = residual_frame.assign(
            absolute_model_error=residual_frame[model_residual_col].abs()
        ).sort_values("absolute_model_error", ascending=False).head(20)
        worst.to_csv(
            TABLE_DIR / f"{target}_worst_20_days.csv", index=False
        )

        plot_actual_vs_predicted(
            common, columns["actual"], methods, target
        )
        plot_residuals(
            common, columns["actual"], methods, target
        )
        plot_monthly_errors(monthly, target)

        importance = load_feature_importance(target)
        if importance is not None:
            importance.to_csv(
                TABLE_DIR / f"{target}_feature_importance.csv", index=False
            )
            save_feature_importance_plot(importance, target)

        report["targets"][target] = {
            "actual_column": columns["actual"],
            "model_prediction_column": columns["model"],
            "persistence_column": columns["persistence"],
            "trend_adjusted_column": columns["trend_adjusted"],
            "test_date_range": {
                "start": common["date"].min().date().isoformat(),
                "end": common["date"].max().date().isoformat(),
            },
            "metrics_same_rows": metrics,
            "demand_range_rows": int(len(ranges)),
            "monthly_rows": int(len(monthly)),
            "worst_20_days_csv": f"tables/{target}_worst_20_days.csv",
            "feature_importance_available": importance is not None,
        }

        model_metrics = metrics["demand_change_model"]
        LOG.info(
            "%s | n=%d | model MAE=%.4f | persistence MAE=%s | "
            "improvement=%s%%",
            target,
            model_metrics["n"],
            model_metrics["mae"],
            (
                f"{metrics['persistence']['mae']:.4f}"
                if "persistence" in metrics else "not available"
            ),
            (
                f"{metrics['improvement_vs_persistence_percent']:.2f}"
                if metrics["improvement_vs_persistence_percent"] is not None
                else "not available"
            ),
        )

    if all_residuals:
        pd.concat(all_residuals, ignore_index=True).to_csv(
            TABLE_DIR / "all_targets_test_residuals.csv", index=False
        )

    report_path = OUTPUT_DIR / "demand_change_diagnostics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    LOG.info("Diagnostics completed.")
    LOG.info("Report: %s", report_path)
    LOG.info("Tables: %s", TABLE_DIR)
    LOG.info("Plots: %s", PLOT_DIR)


if __name__ == "__main__":
    main()
