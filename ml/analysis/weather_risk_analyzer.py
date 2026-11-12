"""
GridGuard AI - Historical Weather Risk Analyzer

Creates explainable, percentile-based weather-risk indicators from the
prior-only historical weather features in the fused dataset.

Important:
- Uses only weather features already present in the fused dataset.
- Thresholds are learned from the training period only (through 2021-02-13).
- Temperature thresholds are month-specific to reduce seasonal bias.
- Weather risk is NOT a grid-overload label or an official weather warning.
- Missing values remain unknown; they are never replaced with zero.
- Location-count diagnostic columns are not used as predictors.
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
    / "weather_risk"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "electricity"
    / "weather_risk"
)

TRAIN_END_DATE = "2021-02-13"

# Each indicator uses a feature already engineered from weather strictly
# before the labelled date. P90/P95 thresholds are calculated from train.
INDICATORS = {
    "heat": {
        "features": [
            "imd_temperature_mean_3d_c_spatial_median",
            "imd_temperature_mean_7d_c_spatial_median",
            "imd_temperature_mean_3d_c_spatial_max",
            "imd_temperature_mean_7d_c_spatial_max",
            "open_meteo_temperature_mean_3d_c_spatial_median",
            "open_meteo_temperature_mean_7d_c_spatial_median",
            "open_meteo_temperature_mean_3d_c_spatial_max",
            "open_meteo_temperature_mean_7d_c_spatial_max",
        ],
        "direction": "high",
        "seasonal_month_thresholds": True,
        "description": "Unusually high recent temperature relative to the same calendar month.",
    },
    "rainfall": {
        "features": [
            "imd_rainfall_3d_mm_spatial_max",
            "imd_rainfall_7d_mm_spatial_max",
            "imd_rainfall_14d_mm_spatial_max",
            "open_meteo_rainfall_3d_mm_spatial_max",
            "open_meteo_rainfall_7d_mm_spatial_max",
            "open_meteo_rainfall_14d_mm_spatial_max",
        ],
        "direction": "high",
        "seasonal_month_thresholds": False,
        "description": "High accumulated rainfall across represented weather locations.",
    },
    "humidity": {
        "features": [
            "open_meteo_humidity_mean_7d_pct_spatial_max",
        ],
        "direction": "high",
        "seasonal_month_thresholds": False,
        "description": "Unusually high recent humidity in represented Open-Meteo locations.",
    },
    "wind": {
        "features": [
            "open_meteo_wind_speed_max_7d_spatial_max",
            "open_meteo_wind_speed_mean_7d_spatial_max",
        ],
        "direction": "high",
        "seasonal_month_thresholds": False,
        "description": "Unusually high recent wind speed in represented Open-Meteo locations.",
    },
    "pressure_drop": {
        "features": [
            "open_meteo_pressure_change_1d_hpa_spatial_median",
            "open_meteo_pressure_change_1d_hpa_spatial_max",
        ],
        "direction": "low",
        "seasonal_month_thresholds": False,
        "description": "Unusually large recent pressure decrease.",
    },
}

PERCENTILES = {"elevated": 90, "high": 95}
CATEGORY_SCORE = {"normal": 0, "elevated": 1, "high": 2, "unknown": np.nan}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("weather_risk_analyzer")


def get_threshold(values: pd.Series, direction: str) -> dict[str, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) < 30:
        return {}

    if direction == "high":
        return {
            "elevated": float(values.quantile(0.90)),
            "high": float(values.quantile(0.95)),
            "observations": int(len(values)),
        }

    return {
        "elevated": float(values.quantile(0.10)),
        "high": float(values.quantile(0.05)),
        "observations": int(len(values)),
    }


def fit_thresholds(
    train: pd.DataFrame,
    available_features: list[str],
) -> dict[str, Any]:
    fitted: dict[str, Any] = {}

    for indicator, config in INDICATORS.items():
        fitted[indicator] = {}

        for feature in config["features"]:
            if feature not in available_features:
                continue

            if config["seasonal_month_thresholds"]:
                month_thresholds = {}
                for month in range(1, 13):
                    subset = train.loc[train["date"].dt.month == month, feature]
                    threshold = get_threshold(subset, config["direction"])
                    if threshold:
                        month_thresholds[str(month)] = threshold
                fitted[indicator][feature] = {
                    "by_month": month_thresholds
                }
            else:
                threshold = get_threshold(
                    train[feature], config["direction"]
                )
                if threshold:
                    fitted[indicator][feature] = {
                        "overall": threshold
                    }

    return fitted


def classify_indicator_value(
    value: Any,
    threshold: dict[str, float],
    direction: str,
) -> str:
    if pd.isna(value) or not threshold:
        return "unknown"

    value = float(value)
    elevated = threshold["elevated"]
    high = threshold["high"]

    if direction == "high":
        if value >= high:
            return "high"
        if value >= elevated:
            return "elevated"
    else:
        if value <= high:
            return "high"
        if value <= elevated:
            return "elevated"

    return "normal"


def feature_threshold(
    fitted: dict[str, Any],
    indicator: str,
    feature: str,
    month: int,
) -> dict[str, float]:
    item = fitted.get(indicator, {}).get(feature, {})
    if "by_month" in item:
        return item["by_month"].get(str(month), {})
    return item.get("overall", {})


def apply_indicators(
    df: pd.DataFrame,
    fitted: dict[str, Any],
    available_features: list[str],
) -> pd.DataFrame:
    result = df.copy()

    for indicator, config in INDICATORS.items():
        feature_labels = []

        for feature in config["features"]:
            if feature not in available_features:
                continue

            labels = []
            for row in result[["date", feature]].itertuples(index=False):
                date_value, raw_value = row
                threshold = feature_threshold(
                    fitted, indicator, feature, int(date_value.month)
                )
                labels.append(
                    classify_indicator_value(
                        raw_value, threshold, config["direction"]
                    )
                )

            label_col = f"{feature}__{indicator}_level"
            result[label_col] = labels
            feature_labels.append(label_col)

        if feature_labels:
            def combine_row(row: pd.Series) -> str:
                labels = [
                    row[col]
                    for col in feature_labels
                    if row[col] in ("normal", "elevated", "high")
                ]
                if not labels:
                    return "unknown"
                if "high" in labels:
                    return "high"
                if "elevated" in labels:
                    return "elevated"
                return "normal"

            result[f"{indicator}_risk"] = result.apply(
                combine_row, axis=1
            )
        else:
            result[f"{indicator}_risk"] = "unknown"

    risk_columns = [f"{name}_risk" for name in INDICATORS]

    def overall_weather_risk(row: pd.Series) -> str:
        values = [row[col] for col in risk_columns]
        known = [value for value in values if value in ("normal", "elevated", "high")]
        if not known:
            return "unknown"
        if "high" in known:
            return "high"
        if "elevated" in known:
            return "elevated"
        return "normal"

    result["weather_risk"] = result.apply(
        overall_weather_risk, axis=1
    )

    result["weather_risk_score"] = result["weather_risk"].map(
        CATEGORY_SCORE
    )

    result["weather_hazard_count"] = result[risk_columns].apply(
        lambda row: sum(value in ("elevated", "high") for value in row),
        axis=1,
    )

    result["weather_high_hazard_count"] = result[risk_columns].apply(
        lambda row: sum(value == "high" for value in row),
        axis=1,
    )

    result["weather_known_indicator_count"] = result[risk_columns].apply(
        lambda row: sum(value in ("normal", "elevated", "high") for value in row),
        axis=1,
    )

    result["weather_coverage"] = (
        result["weather_known_indicator_count"] / len(INDICATORS)
    )

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    logger.info("Loading fused dataset: %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if "date" not in df:
        raise ValueError("Required date column is missing.")

    train_end = pd.Timestamp(TRAIN_END_DATE)
    train = df.loc[df["date"] <= train_end].copy()

    feature_candidates = [
        feature
        for config in INDICATORS.values()
        for feature in config["features"]
    ]
    available_features = [
        feature for feature in feature_candidates if feature in df.columns
    ]

    if not available_features:
        raise ValueError(
            "None of the expected weather features were found. "
            "Check the fused dataset schema."
        )

    missing_features = sorted(set(feature_candidates) - set(available_features))
    logger.info(
        "Weather features found: %d | unavailable: %d",
        len(available_features),
        len(missing_features),
    )
    if missing_features:
        logger.warning("Unavailable features: %s", missing_features)

    fitted = fit_thresholds(train, available_features)
    labelled = apply_indicators(df, fitted, available_features)

    output_csv = OUTPUT_DIR / "historical_weather_risk.csv"
    thresholds_json = REPORT_DIR / "weather_risk_thresholds.json"
    report_json = REPORT_DIR / "weather_risk_report.json"

    labelled.to_csv(output_csv, index=False)

    with thresholds_json.open("w", encoding="utf-8") as file:
        json.dump(fitted, file, indent=4)

    report = {
        "project": "GridGuard AI",
        "module": "historical_weather_risk_analyzer",
        "interpretation": (
            "Relative historical weather-risk indicators. "
            "These are not official meteorological warnings or grid-overload labels."
        ),
        "training_end_date": TRAIN_END_DATE,
        "threshold_method": (
            "Training-period P90/P95 for high-direction indicators; "
            "P10/P5 for pressure-decrease indicators."
        ),
        "temperature_method": (
            "Month-specific training thresholds to reduce seasonal bias."
        ),
        "indicators": {
            name: {
                "description": config["description"],
                "direction": config["direction"],
                "features_used": [
                    feature for feature in config["features"]
                    if feature in available_features
                ],
            }
            for name, config in INDICATORS.items()
        },
        "unavailable_features": missing_features,
        "rows": int(len(labelled)),
        "date_min": str(labelled["date"].min().date()),
        "date_max": str(labelled["date"].max().date()),
        "weather_risk_distribution": {
            str(key): int(value)
            for key, value in labelled["weather_risk"].value_counts(
                dropna=False
            ).to_dict().items()
        },
        "indicator_distributions": {
            f"{name}_risk": {
                str(key): int(value)
                for key, value in labelled[f"{name}_risk"].value_counts(
                    dropna=False
                ).to_dict().items()
            }
            for name in INDICATORS
        },
        "limitations": [
            "Weather features are historical prior-only rolling summaries, not live forecasts.",
            "Weather station coverage is spatially limited and not feeder-level.",
            "A high weather-risk label is a relative statistical indicator, not an official warning.",
            "Weather risk does not establish causation or grid stress.",
            "Missing values remain unknown and are not treated as zero.",
            "Thunderstorm probability and cloud cover are not used.",
        ],
    }

    with report_json.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    logger.info(
        "Weather-risk distribution:\n%s",
        labelled["weather_risk"].value_counts(dropna=False).to_string(),
    )
    logger.info("Saved labelled data: %s", output_csv)
    logger.info("Saved thresholds: %s", thresholds_json)
    logger.info("Saved report: %s", report_json)
    logger.info("Weather-risk analysis completed.")


if __name__ == "__main__":
    main()
