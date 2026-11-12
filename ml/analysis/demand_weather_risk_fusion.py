"""
GridGuard AI - Demand and Weather Risk Fusion

Combines:
- Historical demand-pressure labels from demand_pressure_classifier.py
- Historical weather-risk labels from weather_risk_analyzer.py

This is an explainable screening indicator, NOT a physical grid-overload
prediction. No available-capacity/reserve-margin data is present.

Scoring:
    demand pressure: normal=0, elevated=1, high=2, critical=3
    weather risk:    normal=0, elevated=1, high=2

    combined score = (2 * demand pressure score) + weather risk score

Combined categories:
    0-1: normal
    2-3: elevated
    4-5: high
    6-8: critical

Demand is deliberately weighted twice as much as weather because the
project's primary operational signal is electricity demand. Weather can
increase the screening score, but cannot by itself establish grid stress.

Unknown labels remain unknown for the affected component. The combined
category is unknown if demand pressure is unknown. If weather risk is
unknown, the score is demand-only and the output is explicitly flagged.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEMAND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "demand_pressure"
    / "historical_demand_pressure.csv"
)

WEATHER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weather_risk"
    / "historical_weather_risk.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "risk_fusion"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "risk_fusion"
)

DEMAND_SCORE = {
    "normal": 0,
    "elevated": 1,
    "high": 2,
    "critical": 3,
}

WEATHER_SCORE = {
    "normal": 0,
    "elevated": 1,
    "high": 2,
}

COMBINED_LABELS = {
    0: "normal",
    1: "normal",
    2: "elevated",
    3: "elevated",
    4: "high",
    5: "high",
    6: "critical",
    7: "critical",
    8: "critical",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("demand_weather_risk_fusion")


def load_component(
    path: Path,
    required_columns: set[str],
    component_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{component_name} file not found: {path}")

    frame = pd.read_csv(path, parse_dates=["date"])

    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            f"{component_name} file is missing columns: {sorted(missing)}"
        )

    if frame["date"].duplicated().any():
        duplicates = int(frame["date"].duplicated().sum())
        raise ValueError(
            f"{component_name} contains {duplicates} duplicate dates. "
            "Resolve duplicates before risk fusion."
        )

    return frame


def classify_combined(score: Any) -> str:
    if pd.isna(score):
        return "unknown"
    return COMBINED_LABELS[int(score)]


def build_explanation(row: pd.Series) -> str:
    parts = [
        f"Demand pressure: {row['demand_pressure']}",
        f"Weather risk: {row['weather_risk']}",
    ]

    if row["weather_risk"] == "unknown":
        parts.append("Weather component unavailable; score is demand-only.")

    if row["demand_pressure"] == "critical":
        parts.append("Critical demand-pressure category.")
    elif row["demand_pressure"] == "high":
        parts.append("High demand-pressure category.")
    elif row["demand_pressure"] == "elevated":
        parts.append("Elevated demand-pressure category.")

    high_weather = [
        name
        for name in ("heat", "rainfall", "humidity", "wind", "pressure_drop")
        if row.get(f"{name}_risk") == "high"
    ]
    elevated_weather = [
        name
        for name in ("heat", "rainfall", "humidity", "wind", "pressure_drop")
        if row.get(f"{name}_risk") == "elevated"
    ]

    if high_weather:
        parts.append("High weather indicators: " + ", ".join(high_weather) + ".")
    if elevated_weather:
        parts.append(
            "Elevated weather indicators: "
            + ", ".join(elevated_weather)
            + "."
        )

    return " ".join(parts)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading demand-pressure labels.")
    demand = load_component(
        DEMAND_PATH,
        {"date", "demand_pressure", "energy_pressure", "peak_pressure"},
        "Demand pressure",
    )

    logger.info("Loading weather-risk labels.")
    weather_required = {
        "date",
        "weather_risk",
        "heat_risk",
        "rainfall_risk",
        "humidity_risk",
        "wind_risk",
        "pressure_drop_risk",
    }
    weather = load_component(
        WEATHER_PATH,
        weather_required,
        "Weather risk",
    )

    demand_columns = [
        "date",
        "energy_MU",
        "peak_demand_MW",
        "energy_pressure",
        "peak_pressure",
        "demand_pressure",
        "demand_pressure_score",
        "is_elevated_or_above",
        "is_high_or_critical",
    ]
    demand_columns = [
        column for column in demand_columns if column in demand.columns
    ]

    weather_columns = [
        "date",
        "weather_risk",
        "weather_risk_score",
        "weather_hazard_count",
        "weather_high_hazard_count",
        "weather_known_indicator_count",
        "weather_coverage",
        "heat_risk",
        "rainfall_risk",
        "humidity_risk",
        "wind_risk",
        "pressure_drop_risk",
    ]
    weather_columns = [
        column for column in weather_columns if column in weather.columns
    ]

    fused = demand[demand_columns].merge(
        weather[weather_columns],
        on="date",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = fused["_merge"].value_counts().to_dict()
    fused = fused.drop(columns=["_merge"]).sort_values("date").reset_index(drop=True)

    fused["demand_score_for_fusion"] = fused["demand_pressure"].map(
        DEMAND_SCORE
    )
    fused["weather_score_for_fusion"] = fused["weather_risk"].map(
        WEATHER_SCORE
    )

    # Weather-unknown dates retain a demand-only score, with an explicit flag.
    fused["weather_component_available"] = fused[
        "weather_score_for_fusion"
    ].notna()

    fused["combined_risk_score"] = (
        2 * fused["demand_score_for_fusion"]
        + fused["weather_score_for_fusion"].fillna(0)
    )

    fused["combined_risk"] = fused.apply(
        lambda row: (
            classify_combined(row["combined_risk_score"])
            if pd.notna(row["demand_score_for_fusion"])
            else "unknown"
        ),
        axis=1,
    )

    fused["risk_explanation"] = fused.apply(
        build_explanation, axis=1
    )

    output_csv = OUTPUT_DIR / "historical_combined_risk.csv"
    report_json = REPORT_DIR / "risk_fusion_report.json"

    fused.to_csv(output_csv, index=False)

    report: dict[str, Any] = {
        "project": "GridGuard AI",
        "module": "demand_weather_risk_fusion",
        "interpretation": (
            "Combined statistical screening indicator based on historical "
            "demand pressure and unusual weather conditions. It is not a "
            "verified grid-overload or outage prediction."
        ),
        "formula": (
            "combined_risk_score = 2 * demand_pressure_score + weather_risk_score"
        ),
        "demand_score_mapping": DEMAND_SCORE,
        "weather_score_mapping": WEATHER_SCORE,
        "combined_score_categories": {
            "0-1": "normal",
            "2-3": "elevated",
            "4-5": "high",
            "6-8": "critical",
        },
        "join": {
            "method": "outer join on date",
            "demand_rows": int(len(demand)),
            "weather_rows": int(len(weather)),
            "output_rows": int(len(fused)),
            "date_min": str(fused["date"].min().date()),
            "date_max": str(fused["date"].max().date()),
            "merge_counts": {
                str(key): int(value)
                for key, value in unmatched.items()
            },
        },
        "combined_risk_distribution": {
            str(key): int(value)
            for key, value in fused["combined_risk"].value_counts(
                dropna=False
            ).to_dict().items()
        },
        "weather_component_unavailable_days": int(
            (~fused["weather_component_available"]).sum()
        ),
        "limitations": [
            "No official available-capacity or reserve-margin data is available.",
            "Combined categories are statistical screening levels, not physical grid states.",
            "Weather is weighted less than demand by an explicit design choice.",
            "Weather-only high risk cannot independently establish grid stress.",
            "Historical rolling weather features are not live weather forecasts.",
            "Weather stations do not provide feeder-level coverage.",
            "Missing weather is treated as unavailable, not as zero risk.",
        ],
    }

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info(
        "Combined risk distribution:\n%s",
        fused["combined_risk"].value_counts(dropna=False).to_string(),
    )
    logger.info(
        "Weather component unavailable: %d days",
        report["weather_component_unavailable_days"],
    )
    logger.info("Saved fused risk dataset: %s", output_csv)
    logger.info("Saved report: %s", report_json)
    logger.info("Demand-weather risk fusion completed.")


if __name__ == "__main__":
    main()
