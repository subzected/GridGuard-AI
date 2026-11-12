"""
GridGuard AI - Risk Fusion Validation

Validates the historical demand/weather screening labels.

This script does NOT validate physical grid overload because no official
available-capacity, reserve-margin, or grid-stress labels are available.

It evaluates:
1. Whether demand-only and combined risk categories align with high
   observed daily energy and peak demand.
2. Whether adding weather changes precision/recall and false-alarm rates
   against fixed training-period demand thresholds.
3. How risk labels and high-demand days vary by year and month.
4. How weather indicators are distributed within high-demand observations.
5. Whether combined risk increases alerts without improving high-demand
   identification.

All reference thresholds are loaded from the saved training-only JSON file.
No thresholds or models are fitted in this validation script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FUSED_RISK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_fusion"
    / "historical_combined_risk.csv"
)

DEMAND_THRESHOLDS_PATH = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "demand_pressure"
    / "demand_pressure_thresholds.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "risk_fusion_validation"
)

TABLE_DIR = OUTPUT_DIR / "tables"
PLOT_DIR = OUTPUT_DIR / "plots"

RISK_ORDER = ["normal", "elevated", "high", "critical"]
WEATHER_INDICATORS = [
    "heat_risk",
    "rainfall_risk",
    "humidity_risk",
    "wind_risk",
    "pressure_drop_risk",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("risk_fusion_validation")


def load_inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    if not FUSED_RISK_PATH.exists():
        raise FileNotFoundError(
            f"Risk fusion dataset not found: {FUSED_RISK_PATH}"
        )

    if not DEMAND_THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Demand thresholds not found: {DEMAND_THRESHOLDS_PATH}"
        )

    df = pd.read_csv(FUSED_RISK_PATH, parse_dates=["date"])
    with DEMAND_THRESHOLDS_PATH.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    required = {
        "date",
        "energy_MU",
        "peak_demand_MW",
        "demand_pressure",
        "weather_risk",
        "combined_risk",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Risk fusion dataset missing columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in risk fusion dataset.")

    return df, thresholds


def binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    """Metrics for boolean actual/predicted series, excluding missing values."""
    valid = actual.notna() & predicted.notna()
    y_true = actual.loc[valid].astype(bool)
    y_pred = predicted.loc[valid].astype(bool)

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )

    return {
        "observations": int(valid.sum()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "false_alarm_rate": (
            fp / (fp + tn) if fp + tn else None
        ),
        "alert_rate": float(y_pred.mean()) if len(y_pred) else None,
        "actual_positive_rate": float(y_true.mean()) if len(y_true) else None,
    }


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def add_reference_labels(
    df: pd.DataFrame,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    result = df.copy()

    energy_p90 = float(thresholds["energy"]["p90"])
    energy_p95 = float(thresholds["energy"]["p95"])
    peak_p90 = float(thresholds["peak_demand"]["p90"])
    peak_p95 = float(thresholds["peak_demand"]["p95"])

    result["actual_energy_high"] = np.where(
        result["energy_MU"].notna(),
        result["energy_MU"] >= energy_p90,
        np.nan,
    )
    result["actual_energy_critical"] = np.where(
        result["energy_MU"].notna(),
        result["energy_MU"] >= energy_p95,
        np.nan,
    )
    result["actual_peak_high"] = np.where(
        result["peak_demand_MW"].notna(),
        result["peak_demand_MW"] >= peak_p90,
        np.nan,
    )
    result["actual_peak_critical"] = np.where(
        result["peak_demand_MW"].notna(),
        result["peak_demand_MW"] >= peak_p95,
        np.nan,
    )

    result["demand_alert"] = result["demand_pressure"].isin(
        ["high", "critical"]
    )
    result["combined_alert"] = result["combined_risk"].isin(
        ["high", "critical"]
    )
    result["demand_critical_alert"] = (
        result["demand_pressure"] == "critical"
    )
    result["combined_critical_alert"] = (
        result["combined_risk"] == "critical"
    )

    return result


def build_performance_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    reference_targets = {
        "energy_p90": "actual_energy_high",
        "energy_p95": "actual_energy_critical",
        "peak_p90": "actual_peak_high",
        "peak_p95": "actual_peak_critical",
    }

    predictions = {
        "demand_only_high_or_critical": "demand_alert",
        "combined_high_or_critical": "combined_alert",
        "demand_only_critical": "demand_critical_alert",
        "combined_critical": "combined_critical_alert",
    }

    for reference_name, actual_col in reference_targets.items():
        for prediction_name, pred_col in predictions.items():
            metrics = binary_metrics(df[actual_col], df[pred_col])
            rows.append({
                "reference": reference_name,
                "prediction": prediction_name,
                **metrics,
            })

    return pd.DataFrame(rows)


def build_yearly_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[df["combined_risk"] != "unknown"].copy()
    valid["year"] = valid["date"].dt.year

    rows = []
    for year, group in valid.groupby("year"):
        row: dict[str, Any] = {
            "year": int(year),
            "observations": int(len(group)),
            "mean_energy_MU": float(group["energy_MU"].mean()),
            "mean_peak_demand_MW": float(group["peak_demand_MW"].mean()),
            "high_or_critical_demand_pct": float(
                group["demand_pressure"].isin(["high", "critical"]).mean() * 100
            ),
            "high_or_critical_combined_pct": float(
                group["combined_risk"].isin(["high", "critical"]).mean() * 100
            ),
            "critical_combined_pct": float(
                (group["combined_risk"] == "critical").mean() * 100
            ),
            "high_weather_pct": float(
                (group["weather_risk"] == "high").mean() * 100
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def build_monthly_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[df["combined_risk"] != "unknown"].copy()
    valid["month"] = valid["date"].dt.month

    rows = []
    for month, group in valid.groupby("month"):
        rows.append({
            "month": int(month),
            "observations": int(len(group)),
            "mean_energy_MU": float(group["energy_MU"].mean()),
            "mean_peak_demand_MW": float(group["peak_demand_MW"].mean()),
            "high_or_critical_demand_pct": float(
                group["demand_pressure"].isin(["high", "critical"]).mean() * 100
            ),
            "high_or_critical_combined_pct": float(
                group["combined_risk"].isin(["high", "critical"]).mean() * 100
            ),
            "high_weather_pct": float(
                (group["weather_risk"] == "high").mean() * 100
            ),
        })

    return pd.DataFrame(rows)


def build_weather_association_table(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[
        df["actual_peak_high"].notna()
        & df["demand_pressure"].notna()
    ].copy()
    valid["actual_peak_high"] = valid["actual_peak_high"].astype(bool)

    rows = []
    for indicator in WEATHER_INDICATORS:
        if indicator not in valid.columns:
            continue

        for level in ("elevated", "high"):
            selected = valid[indicator] == level
            high_peak = valid["actual_peak_high"]

            rows.append({
                "indicator": indicator,
                "level": level,
                "days_at_level": int(selected.sum()),
                "high_peak_days_at_level": int(
                    (selected & high_peak).sum()
                ),
                "share_of_level_days_with_peak_p90": (
                    float(high_peak[selected].mean() * 100)
                    if selected.any() else None
                ),
                "share_of_all_peak_p90_days_at_level": (
                    float(selected[high_peak].mean() * 100)
                    if high_peak.any() else None
                ),
            })

    return pd.DataFrame(rows)


def build_cross_tab(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.loc[
        df["demand_pressure"].isin(RISK_ORDER)
        & df["weather_risk"].isin(["normal", "elevated", "high"])
    ]

    table = pd.crosstab(
        valid["demand_pressure"],
        valid["weather_risk"],
        normalize="index",
    ) * 100

    table = table.reindex(
        index=RISK_ORDER,
        columns=["normal", "elevated", "high"],
        fill_value=0,
    )
    return table.reset_index(names="demand_pressure")


def save_plots(
    df: pd.DataFrame,
    yearly: pd.DataFrame,
    monthly: pd.DataFrame,
    cross_tab: pd.DataFrame,
) -> None:
    # 1. Yearly combined vs demand-only alert rates.
    if not yearly.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(yearly))
        width = 0.36
        ax.bar(
            x - width / 2,
            yearly["high_or_critical_demand_pct"],
            width,
            label="Demand-only high/critical",
        )
        ax.bar(
            x + width / 2,
            yearly["high_or_critical_combined_pct"],
            width,
            label="Combined high/critical",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(yearly["year"].astype(str), rotation=45)
        ax.set_ylabel("Days (%)")
        ax.set_title("Yearly high/critical alert frequency")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "yearly_alert_rates.png", dpi=160)
        plt.close(fig)

    # 2. Monthly seasonality.
    if not monthly.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(
            monthly["month"],
            monthly["high_or_critical_demand_pct"],
            marker="o",
            label="Demand-only high/critical",
        )
        ax.plot(
            monthly["month"],
            monthly["high_or_critical_combined_pct"],
            marker="o",
            label="Combined high/critical",
        )
        ax.set_xticks(range(1, 13))
        ax.set_xlabel("Month")
        ax.set_ylabel("Days (%)")
        ax.set_title("Monthly high/critical alert frequency")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "monthly_alert_rates.png", dpi=160)
        plt.close(fig)

    # 3. Demand-pressure vs weather-risk cross-tab.
    if not cross_tab.empty:
        plot_df = cross_tab.set_index("demand_pressure")
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_df.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Share within demand category (%)")
        ax.set_xlabel("Demand pressure")
        ax.set_title("Weather-risk distribution within demand categories")
        ax.legend(title="Weather risk")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "demand_weather_crosstab.png", dpi=160)
        plt.close(fig)

    # 4. Actual peak demand and risk labels over time.
    valid = df.loc[df["peak_demand_MW"].notna()].copy()
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(13, 5))
        ax.plot(
            valid["date"],
            valid["peak_demand_MW"],
            linewidth=0.8,
            label="Actual peak demand",
        )
        ax.set_ylabel("Peak demand (MW)")
        ax.set_title("Actual peak demand over time")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "actual_peak_demand_timeline.png", dpi=160)
        plt.close(fig)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading risk fusion dataset and training thresholds.")
    df, thresholds = load_inputs()
    df = add_reference_labels(df, thresholds)

    performance = build_performance_table(df)
    yearly = build_yearly_table(df)
    monthly = build_monthly_table(df)
    weather_assoc = build_weather_association_table(df)
    cross_tab = build_cross_tab(df)

    performance.to_csv(
        TABLE_DIR / "risk_reference_performance.csv", index=False
    )
    yearly.to_csv(TABLE_DIR / "yearly_risk_stability.csv", index=False)
    monthly.to_csv(TABLE_DIR / "monthly_risk_stability.csv", index=False)
    weather_assoc.to_csv(
        TABLE_DIR / "weather_indicator_high_peak_association.csv",
        index=False,
    )
    cross_tab.to_csv(
        TABLE_DIR / "demand_weather_crosstab_pct.csv",
        index=False,
    )

    save_plots(df, yearly, monthly, cross_tab)

    # Key comparison: high/critical combined alerts vs high/critical demand.
    peak_reference = df["actual_peak_high"]
    demand_metrics = binary_metrics(peak_reference, df["demand_alert"])
    combined_metrics = binary_metrics(peak_reference, df["combined_alert"])

    energy_reference = df["actual_energy_high"]
    demand_energy_metrics = binary_metrics(
        energy_reference, df["demand_alert"]
    )
    combined_energy_metrics = binary_metrics(
        energy_reference, df["combined_alert"]
    )

    summary = {
        "project": "GridGuard AI",
        "module": "risk_fusion_validation",
        "validation_scope": (
            "Historical statistical consistency against actual demand thresholds. "
            "This is not validation against physical grid-overload labels."
        ),
        "threshold_source": str(DEMAND_THRESHOLDS_PATH),
        "reference_thresholds": {
            "energy_p90_MU": thresholds["energy"]["p90"],
            "energy_p95_MU": thresholds["energy"]["p95"],
            "peak_p90_MW": thresholds["peak_demand"]["p90"],
            "peak_p95_MW": thresholds["peak_demand"]["p95"],
        },
        "rows": int(len(df)),
        "valid_energy_rows": int(df["energy_MU"].notna().sum()),
        "valid_peak_rows": int(df["peak_demand_MW"].notna().sum()),
        "peak_p90_reference": {
            "demand_only": demand_metrics,
            "combined": combined_metrics,
        },
        "energy_p90_reference": {
            "demand_only": demand_energy_metrics,
            "combined": combined_energy_metrics,
        },
        "critical_note": (
            "The P90/P95 demand thresholds are statistical high-demand references. "
            "They are not capacity limits, reserve margins, or official grid-stress labels."
        ),
        "interpretation_limits": [
            "Demand-only labels are derived from actual demand and are not independent ground truth.",
            "Combined labels include demand pressure, so comparison with demand thresholds is a consistency check, not an independent model validation.",
            "Weather association is observational and does not establish causality.",
            "No physical grid capacity, reserve margin, outage, or operator stress labels are available.",
            "High combined-risk alert rates should be reviewed for false alarms before operational use.",
        ],
    }

    report_path = OUTPUT_DIR / "risk_fusion_validation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    logger.info("Peak P90 reference comparison:")
    logger.info(
        "Demand-only | precision=%s recall=%s false_alarm_rate=%s alert_rate=%s",
        demand_metrics["precision"],
        demand_metrics["recall"],
        demand_metrics["false_alarm_rate"],
        demand_metrics["alert_rate"],
    )
    logger.info(
        "Combined     | precision=%s recall=%s false_alarm_rate=%s alert_rate=%s",
        combined_metrics["precision"],
        combined_metrics["recall"],
        combined_metrics["false_alarm_rate"],
        combined_metrics["alert_rate"],
    )
    logger.info("Saved validation report: %s", report_path)
    logger.info("Saved tables: %s", TABLE_DIR)
    logger.info("Saved plots: %s", PLOT_DIR)
    logger.info("Risk-fusion validation completed.")


if __name__ == "__main__":
    main()
