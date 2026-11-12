"""
GridGuard AI - Risk Assessment Engine

Purpose:
- Keep demand pressure as the primary operational screening category.
- Keep weather hazards as independent contextual indicators.
- Identify heat-demand coincidence without claiming causation.
- Generate transparent daily explanations and dashboard-ready outputs.

Input:
    data/processed/risk_fusion/historical_combined_risk.csv

Outputs:
    data/processed/risk_assessment/historical_risk_assessment.csv
    reports/electricity/risk_assessment/risk_assessment_report.json
    reports/electricity/risk_assessment/tables/...

Important limitations:
- This is a statistical screening/assessment layer, not a physical grid
  overload, outage, or capacity adequacy detector.
- No available-capacity or reserve-margin data is available.
- Weather indicators are historical prior-only summaries, not live forecasts.
- Heat-demand coincidence is an association flag, not a causal claim.
- The engine does not escalate demand pressure because of weather alone.
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
    / "risk_fusion"
    / "historical_combined_risk.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_assessment"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "risk_assessment"
)

TABLE_DIR = REPORT_DIR / "tables"

DEMAND_ORDER = {
    "normal": 0,
    "elevated": 1,
    "high": 2,
    "critical": 3,
}

WEATHER_ORDER = {
    "normal": 0,
    "elevated": 1,
    "high": 2,
}

WEATHER_COLUMNS = {
    "heat": "heat_risk",
    "rainfall": "rainfall_risk",
    "humidity": "humidity_risk",
    "wind": "wind_risk",
    "pressure_drop": "pressure_drop_risk",
}

# Demand categories considered elevated for coincidence analysis.
ELEVATED_DEMAND_CATEGORIES = {"elevated", "high", "critical"}
HIGH_DEMAND_CATEGORIES = {"high", "critical"}
ELEVATED_HEAT_CATEGORIES = {"elevated", "high"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("risk_assessment_engine")


def load_input() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Risk-fusion dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])

    required = {
        "date",
        "energy_MU",
        "peak_demand_MW",
        "energy_pressure",
        "peak_pressure",
        "demand_pressure",
        "weather_risk",
        "heat_risk",
        "rainfall_risk",
        "humidity_risk",
        "wind_risk",
        "pressure_drop_risk",
    }

    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Input dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("Input contains invalid or missing dates.")

    if df["date"].duplicated().any():
        count = int(df["date"].duplicated().sum())
        raise ValueError(
            f"Input contains {count} duplicate dates. "
            "Resolve duplicates before risk assessment."
        )

    return df


def validate_categories(df: pd.DataFrame) -> None:
    allowed_demand = set(DEMAND_ORDER) | {"unknown"}
    allowed_weather = set(WEATHER_ORDER) | {"unknown"}

    for column in ("energy_pressure", "peak_pressure", "demand_pressure"):
        unexpected = set(df[column].dropna().astype(str)) - allowed_demand
        if unexpected:
            raise ValueError(
                f"Unexpected categories in {column}: {sorted(unexpected)}"
            )

    for column in WEATHER_COLUMNS.values():
        unexpected = set(df[column].dropna().astype(str)) - allowed_weather
        if unexpected:
            raise ValueError(
                f"Unexpected categories in {column}: {sorted(unexpected)}"
            )


def combine_weather_context(row: pd.Series) -> str:
    values = [
        row[column]
        for column in WEATHER_COLUMNS.values()
        if row[column] in WEATHER_ORDER
    ]

    if not values:
        return "unknown"
    if "high" in values:
        return "high"
    if "elevated" in values:
        return "elevated"
    return "normal"


def build_assessments(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # The primary assessment is demand pressure only.
    result["primary_risk_category"] = result["demand_pressure"].where(
        result["demand_pressure"].isin(DEMAND_ORDER),
        "unknown",
    )

    result["primary_risk_score"] = result[
        "primary_risk_category"
    ].map(DEMAND_ORDER)

    # Preserve weather as a separate context field; do not escalate demand.
    result["weather_context_category"] = result.apply(
        combine_weather_context, axis=1
    )

    result["weather_watch"] = result["weather_context_category"].isin(
        ["elevated", "high"]
    )

    # Heat-demand coincidence: flags overlap, not causation.
    result["heat_demand_coincidence"] = (
        result["demand_pressure"].isin(ELEVATED_DEMAND_CATEGORIES)
        & result["heat_risk"].isin(ELEVATED_HEAT_CATEGORIES)
    )

    result["high_demand_heat_coincidence"] = (
        result["demand_pressure"].isin(HIGH_DEMAND_CATEGORIES)
        & result["heat_risk"].isin(ELEVATED_HEAT_CATEGORIES)
    )

    # Additional independent coincidence flags for dashboard filtering.
    for indicator, column in WEATHER_COLUMNS.items():
        result[f"{indicator}_demand_coincidence"] = (
            result["demand_pressure"].isin(ELEVATED_DEMAND_CATEGORIES)
            & result[column].isin(ELEVATED_HEAT_CATEGORIES)
        )

    # Transparent demand sub-signal: the more severe of energy/peak pressure.
    result["energy_peak_agreement"] = np.select(
        [
            result["energy_pressure"] == result["peak_pressure"],
            result["energy_pressure"].isin(DEMAND_ORDER)
            & result["peak_pressure"].isin(DEMAND_ORDER),
        ],
        [
            "same_category",
            "different_categories",
        ],
        default="unknown",
    )

    result["demand_signal_detail"] = np.select(
        [
            result["energy_pressure"].isin(DEMAND_ORDER)
            & result["peak_pressure"].isin(DEMAND_ORDER),
            result["energy_pressure"].isin(DEMAND_ORDER),
            result["peak_pressure"].isin(DEMAND_ORDER),
        ],
        [
            "energy_and_peak_available",
            "energy_only_available",
            "peak_only_available",
        ],
        default="demand_unavailable",
    )

    result["risk_assessment_explanation"] = result.apply(
        build_explanation, axis=1
    )

    result["assessment_status"] = np.select(
        [
            result["primary_risk_category"].eq("unknown"),
            result["weather_context_category"].eq("unknown"),
        ],
        [
            "demand_unavailable",
            "demand_assessed_weather_incomplete",
        ],
        default="demand_and_weather_assessed",
    )

    return result


def build_explanation(row: pd.Series) -> str:
    demand = row["demand_pressure"]
    weather = row["weather_context_category"]

    if demand == "unknown":
        parts = ["Demand pressure is unavailable because demand targets are missing."]
    else:
        parts = [f"Primary demand-pressure category is {demand}."]

    if weather == "unknown":
        parts.append("Weather context is unavailable or incomplete.")
    else:
        parts.append(f"Overall weather context is {weather}.")

    if row["heat_demand_coincidence"]:
        parts.append(
            "Elevated demand coincides with elevated/high heat indicators; "
            "this is an association flag, not evidence of causation."
        )
    elif row["heat_risk"] in ELEVATED_HEAT_CATEGORIES:
        parts.append(
            "Elevated/high heat indicators are present without elevated "
            "demand pressure."
        )

    active_hazards = [
        name
        for name, column in WEATHER_COLUMNS.items()
        if row[column] in ELEVATED_HEAT_CATEGORIES
    ]

    if active_hazards:
        parts.append(
            "Weather indicators to review: " + ", ".join(active_hazards) + "."
        )

    if demand == "critical":
        parts.append(
            "Critical is a historical percentile-based demand category, "
            "not a verified grid-overload state."
        )

    return " ".join(parts)


def build_indicator_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for indicator, column in WEATHER_COLUMNS.items():
        for level in ("normal", "elevated", "high", "unknown"):
            subset = df.loc[df[column] == level]
            if subset.empty:
                continue

            rows.append({
                "indicator": indicator,
                "weather_level": level,
                "days": int(len(subset)),
                "elevated_or_above_demand_pct": float(
                    subset["demand_pressure"].isin(
                        ELEVATED_DEMAND_CATEGORIES
                    ).mean() * 100
                ),
                "high_or_critical_demand_pct": float(
                    subset["demand_pressure"].isin(
                        HIGH_DEMAND_CATEGORIES
                    ).mean() * 100
                ),
                "mean_energy_MU": (
                    float(subset["energy_MU"].mean())
                    if subset["energy_MU"].notna().any()
                    else None
                ),
                "mean_peak_demand_MW": (
                    float(subset["peak_demand_MW"].mean())
                    if subset["peak_demand_MW"].notna().any()
                    else None
                ),
            })

    return pd.DataFrame(rows)


def build_yearly_summary(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["year"] = frame["date"].dt.year

    rows = []
    for year, group in frame.groupby("year"):
        valid_demand = group["primary_risk_category"].isin(DEMAND_ORDER)

        rows.append({
            "year": int(year),
            "calendar_days": int(len(group)),
            "valid_demand_days": int(valid_demand.sum()),
            "normal_demand_pct": float(
                (group.loc[valid_demand, "demand_pressure"] == "normal").mean() * 100
            ) if valid_demand.any() else None,
            "elevated_or_above_demand_pct": float(
                group.loc[valid_demand, "demand_pressure"].isin(
                    ELEVATED_DEMAND_CATEGORIES
                ).mean() * 100
            ) if valid_demand.any() else None,
            "high_or_critical_demand_pct": float(
                group.loc[valid_demand, "demand_pressure"].isin(
                    HIGH_DEMAND_CATEGORIES
                ).mean() * 100
            ) if valid_demand.any() else None,
            "heat_watch_pct": float(
                group["heat_risk"].isin(ELEVATED_HEAT_CATEGORIES).mean() * 100
            ),
            "heat_demand_coincidence_days": int(
                group["heat_demand_coincidence"].sum()
            ),
            "mean_peak_demand_MW": (
                float(group["peak_demand_MW"].mean())
                if group["peak_demand_MW"].notna().any()
                else None
            ),
        })

    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame) -> dict[str, Any]:
    valid_demand = df["demand_pressure"].isin(DEMAND_ORDER)
    valid_weather = df["weather_context_category"].isin(WEATHER_ORDER)

    demand_counts = (
        df["primary_risk_category"].value_counts(dropna=False).to_dict()
    )
    weather_counts = (
        df["weather_context_category"].value_counts(dropna=False).to_dict()
    )

    return {
        "project": "GridGuard AI",
        "module": "risk_assessment_engine",
        "design": {
            "primary_signal": "historical demand pressure",
            "secondary_signal": "independent weather context",
            "weather_escalates_demand_category": False,
            "heat_demand_coincidence_is_causal": False,
        },
        "dataset": {
            "rows": int(len(df)),
            "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()),
            "valid_demand_days": int(valid_demand.sum()),
            "valid_weather_context_days": int(valid_weather.sum()),
        },
        "primary_risk_distribution": {
            str(key): int(value) for key, value in demand_counts.items()
        },
        "weather_context_distribution": {
            str(key): int(value) for key, value in weather_counts.items()
        },
        "coincidence": {
            "elevated_demand_and_elevated_or_high_heat_days": int(
                df["heat_demand_coincidence"].sum()
            ),
            "high_or_critical_demand_and_elevated_or_high_heat_days": int(
                df["high_demand_heat_coincidence"].sum()
            ),
            "heat_demand_coincidence_rate_among_valid_demand_days_pct": (
                float(
                    df.loc[valid_demand, "heat_demand_coincidence"].mean() * 100
                )
                if valid_demand.any()
                else None
            ),
        },
        "limitations": [
            "No available-capacity, reserve-margin, or official grid-stress labels are available.",
            "Demand categories are statistical historical percentile categories.",
            "Weather is contextual and does not automatically escalate demand pressure.",
            "Heat-demand coincidence is observational and does not prove causation.",
            "Weather features are historical prior-only summaries, not live forecasts.",
            "Weather locations are not feeder-level measurements.",
        ],
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading historical demand-weather risk dataset.")
    df = load_input()
    validate_categories(df)

    logger.info("Building independent demand and weather assessments.")
    assessed = build_assessments(df)

    indicator_summary = build_indicator_summary(assessed)
    yearly_summary = build_yearly_summary(assessed)
    report = build_report(assessed)

    output_csv = OUTPUT_DIR / "historical_risk_assessment.csv"
    report_json = REPORT_DIR / "risk_assessment_report.json"

    assessed.to_csv(output_csv, index=False)
    indicator_summary.to_csv(
        TABLE_DIR / "weather_indicator_demand_summary.csv",
        index=False,
    )
    yearly_summary.to_csv(
        TABLE_DIR / "yearly_risk_assessment_summary.csv",
        index=False,
    )

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info(
        "Primary demand-risk distribution:\n%s",
        assessed["primary_risk_category"].value_counts(dropna=False).to_string(),
    )
    logger.info(
        "Weather-context distribution:\n%s",
        assessed["weather_context_category"].value_counts(dropna=False).to_string(),
    )
    logger.info(
        "Elevated-demand + elevated/high-heat coincidence days: %d",
        int(assessed["heat_demand_coincidence"].sum()),
    )
    logger.info(
        "High/critical-demand + elevated/high-heat coincidence days: %d",
        int(assessed["high_demand_heat_coincidence"].sum()),
    )
    logger.info("Saved daily assessment: %s", output_csv)
    logger.info("Saved report: %s", report_json)
    logger.info("Saved tables: %s", TABLE_DIR)
    logger.info("Risk assessment engine completed.")


if __name__ == "__main__":
    main()
