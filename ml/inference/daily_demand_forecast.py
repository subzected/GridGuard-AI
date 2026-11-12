"""
GridGuard AI — One-day-ahead operational demand inference.

Loads the selected demand-change model bundles:
  - energy_MU: LightGBM delta model
  - peak_demand_MW: XGBoost delta model

The models predict:
    delta(t) = demand(t) - demand(t-1)
    demand_hat(t) = observed_demand(t-1) + predicted_delta(t)

IMPORTANT:
- The input fused dataset must be refreshed with the latest observed daily
  electricity data before this is used operationally.
- The trained bundles expect the exact 96 training features, including
  historical weather features. For a first inference run, the most recent
  available historical weather feature vector is carried forward. This is
  explicitly recorded as stale/carry-forward context, NOT a weather forecast.
- No future weather-risk category is invented. It remains "not_assessed"
  until a point-in-time weather forecast integration is implemented.
- Demand-pressure labels are statistical percentiles, not physical overload.

Run from project root:
    python -m ml.inference.daily_demand_forecast

For a historical/demo run against old data only:
    python -m ml.inference.daily_demand_forecast --allow-stale-history

Optional:
    python -m ml.inference.daily_demand_forecast --forecast-date 2024-04-29
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
)
MODEL_DIR = ROOT / "ml" / "models" / "demand_change"
THRESHOLDS_PATH = (
    ROOT / "reports" / "electricity" / "demand_pressure"
    / "demand_pressure_thresholds.json"
)
OUTPUT_DIR = ROOT / "data" / "processed" / "operational_forecasts"
REPORT_DIR = ROOT / "reports" / "electricity" / "operational_forecasts"

TARGETS = ("energy_MU", "peak_demand_MW")
BAD_FEATURES = {
    "imd_locations_with_any_weather_feature",
    "open_meteo_locations_with_any_weather_feature",
}
LAG_DAYS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 30)
MAX_HISTORY_AGE_DAYS = 7

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
logger = logging.getLogger("gridguard_daily_inference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a one-day-ahead GridGuard AI demand forecast."
    )
    parser.add_argument(
        "--forecast-date",
        type=str,
        default=None,
        help="Optional target date (YYYY-MM-DD). Must be exactly one day after latest history.",
    )
    parser.add_argument(
        "--allow-stale-history",
        action="store_true",
        help="Allow a historical/demo forecast when latest demand data is older than 7 days.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DATASET_PATH),
        help="Path to the refreshed fused daily demand-weather CSV.",
    )
    return parser.parse_args()


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {path}\n"
            "Build/refresh the fused demand-weather dataset first."
        )

    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    if df["date"].isna().any():
        raise ValueError("Dataset contains invalid dates.")
    if df["date"].duplicated().any():
        raise ValueError("Dataset contains duplicate dates.")

    df = df.sort_values("date").set_index("date")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    if not df.index.equals(full_index):
        raise ValueError(
            "Dataset dates are not a complete daily calendar. "
            "Do not calculate day lags across missing calendar dates. "
            "Rebuild the fused dataset with the full daily calendar first."
        )

    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    if df[list(TARGETS)].dropna().empty:
        raise ValueError("No valid demand observations are available.")

    return df


def calendar_features(forecast_date: pd.Timestamp) -> dict[str, float]:
    day_of_year = int(forecast_date.dayofyear)
    return {
        "day_of_week": float(forecast_date.dayofweek),
        "day_of_month": float(forecast_date.day),
        "month": float(forecast_date.month),
        "day_of_year": float(day_of_year),
        "week_of_year": float(forecast_date.isocalendar().week),
        "is_weekend": float(forecast_date.dayofweek >= 5),
        "day_of_year_sin": float(np.sin(2 * np.pi * day_of_year / 365.25)),
        "day_of_year_cos": float(np.cos(2 * np.pi * day_of_year / 365.25)),
    }


def _value_on_date(series: pd.Series, date: pd.Timestamp) -> float:
    if date not in series.index:
        return np.nan
    value = series.loc[date]
    return float(value) if pd.notna(value) else np.nan


def build_demand_features(
    history: pd.DataFrame,
    forecast_date: pd.Timestamp,
) -> dict[str, float]:
    """Recreate the training demand feature definitions for forecast_date."""
    row: dict[str, float] = calendar_features(forecast_date)

    for target in TARGETS:
        series = history[target].astype(float)

        for lag in LAG_DAYS:
            lag_date = forecast_date - pd.Timedelta(days=lag)
            row[f"{target}_lag_{lag}d"] = _value_on_date(series, lag_date)

        prior = series.loc[series.index < forecast_date]
        for window in ROLLING_WINDOWS:
            # Full calendar window immediately before the forecast date.
            window_dates = pd.date_range(
                forecast_date - pd.Timedelta(days=window),
                forecast_date - pd.Timedelta(days=1),
                freq="D",
            )
            values = series.reindex(window_dates)
            if len(values) == window and values.notna().all():
                row[f"{target}_rolling_mean_{window}d"] = float(values.mean())
                row[f"{target}_rolling_std_{window}d"] = float(values.std(ddof=1))
                row[f"{target}_rolling_min_{window}d"] = float(values.min())
                row[f"{target}_rolling_max_{window}d"] = float(values.max())
            else:
                row[f"{target}_rolling_mean_{window}d"] = np.nan
                row[f"{target}_rolling_std_{window}d"] = np.nan
                row[f"{target}_rolling_min_{window}d"] = np.nan
                row[f"{target}_rolling_max_{window}d"] = np.nan

        # Match the original feature builder: pct_change on past.shift(1).
        previous = prior
        if len(previous) >= 2:
            a, b = previous.iloc[-1], previous.iloc[-2]
            row[f"{target}_change_1d_pct"] = (
                float((a / b - 1.0) * 100.0)
                if pd.notna(a) and pd.notna(b) and b != 0
                else np.nan
            )
        else:
            row[f"{target}_change_1d_pct"] = np.nan

        if len(previous) >= 8:
            a, b = previous.iloc[-1], previous.iloc[-8]
            row[f"{target}_change_7d_pct"] = (
                float((a / b - 1.0) * 100.0)
                if pd.notna(a) and pd.notna(b) and b != 0
                else np.nan
            )
        else:
            row[f"{target}_change_7d_pct"] = np.nan

    return row


def load_model_bundle(target: str) -> dict[str, Any]:
    if target == "energy_MU":
        path = MODEL_DIR / "energy_mu_lightgbm_delta.joblib"
    elif target == "peak_demand_MW":
        path = MODEL_DIR / "peak_demand_mw_xgboost_delta.joblib"
    else:
        raise ValueError(f"Unsupported target: {target}")

    if not path.exists():
        raise FileNotFoundError(
            f"Trained model bundle not found: {path}\n"
            "Run python -m ml.training.train_demand_change_models first."
        )

    bundle = joblib.load(path)
    required = {"pipeline", "target", "model_name", "feature_columns"}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f"Model bundle {path.name} missing keys: {sorted(missing)}")
    if bundle["target"] != target:
        raise ValueError(
            f"Model bundle target mismatch: expected {target}, got {bundle['target']}"
        )
    return bundle


def load_thresholds() -> dict[str, Any]:
    if not THRESHOLDS_PATH.exists():
        raise FileNotFoundError(
            f"Demand thresholds not found: {THRESHOLDS_PATH}\n"
            "Run python -m ml.analysis.demand_pressure_classifier first."
        )
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as file:
        thresholds = json.load(file)

    for key in ("energy", "peak_demand"):
        if key not in thresholds:
            raise ValueError(f"Threshold JSON missing '{key}' section.")
        for percentile in ("p80", "p90", "p95"):
            if percentile not in thresholds[key]:
                raise ValueError(f"Threshold JSON missing {key}.{percentile}")
    return thresholds


def classify_demand(value: float, threshold: dict[str, float]) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value >= threshold["p95"]:
        return "critical"
    if value >= threshold["p90"]:
        return "high"
    if value >= threshold["p80"]:
        return "elevated"
    return "normal"


def combine_demand_categories(energy_category: str, peak_category: str) -> str:
    known = [
        category for category in (energy_category, peak_category)
        if category in CATEGORY_ORDER
    ]
    if not known:
        return "unknown"
    return max(known, key=lambda category: CATEGORY_ORDER[category])


def make_forecast_feature_row(
    history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    generated = build_demand_features(history, forecast_date)
    latest = history.iloc[-1]

    # Keep the exact feature contract from the trained artifact. Historical
    # weather features are carried forward from the latest available row only
    # because the model was trained with them. They are not future forecasts.
    weather_columns = [
        col for col in feature_columns
        if col not in generated and col in history.columns
    ]
    for col in weather_columns:
        generated[col] = (
            float(latest[col]) if pd.notna(latest[col]) else np.nan
        )

    missing = [col for col in feature_columns if col not in generated]
    if missing:
        raise ValueError(
            "Cannot construct all model features. Missing columns: "
            f"{missing[:25]}{' ...' if len(missing) > 25 else ''}"
        )

    X = pd.DataFrame(
        [{col: generated[col] for col in feature_columns}],
        columns=feature_columns,
    )
    X = X.apply(pd.to_numeric, errors="coerce")

    missing_features = [
        col for col in feature_columns if pd.isna(X.iloc[0][col])
    ]
    weather_non_null = int(
        X[weather_columns].notna().sum(axis=1).iloc[0]
    ) if weather_columns else 0

    metadata = {
        "feature_count": len(feature_columns),
        "missing_feature_count_before_model_imputation": len(missing_features),
        "missing_feature_names": missing_features,
        "historical_weather_feature_count": len(weather_columns),
        "historical_weather_non_null_count": weather_non_null,
        "weather_feature_policy": (
            "latest_available_historical_feature_row_carried_forward; "
            "not a forecast"
        ),
        "weather_feature_source_date": str(history.index[-1].date()),
    }
    return X, metadata


def forecast_target(
    history: pd.DataFrame,
    forecast_date: pd.Timestamp,
    target: str,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_model_bundle(target)
    feature_columns = list(bundle["feature_columns"])

    X, feature_metadata = make_forecast_feature_row(
        history, forecast_date, feature_columns
    )

    lag_date = forecast_date - pd.Timedelta(days=1)
    lag_value = _value_on_date(history[target], lag_date)
    if not np.isfinite(lag_value):
        raise ValueError(
            f"Cannot forecast {target}: previous calendar day's observed value "
            f"is missing for {lag_date.date()}."
        )

    predicted_delta = float(bundle["pipeline"].predict(X)[0])
    predicted_level = max(0.0, lag_value + predicted_delta)

    threshold_key = "energy" if target == "energy_MU" else "peak_demand"
    category = classify_demand(predicted_level, thresholds[threshold_key])

    result = {
        "target": target,
        "unit": "MU" if target == "energy_MU" else "MW",
        "forecast_value": round(predicted_level, 3),
        "previous_day_actual": round(lag_value, 3),
        "predicted_change": round(predicted_delta, 3),
        "predicted_change_pct": (
            round((predicted_delta / lag_value) * 100.0, 3)
            if lag_value != 0 else None
        ),
        "demand_pressure_category": category,
        "model_name": bundle["model_name"],
        "model_file": (
            f"{target.lower()}_"
            f"{'lightgbm' if target == 'energy_MU' else 'xgboost'}_delta.joblib"
        ),
    }
    return result, feature_metadata


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading refreshed daily demand-weather dataset.")
    history = load_history(input_path)
    latest_date = history.index[-1]
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date())

    if args.forecast_date:
        forecast_date = pd.Timestamp(args.forecast_date).normalize()
    else:
        forecast_date = latest_date + pd.Timedelta(days=1)

    if forecast_date != latest_date + pd.Timedelta(days=1):
        raise ValueError(
            "This module supports one-day-ahead forecasting only. "
            f"Expected { (latest_date + pd.Timedelta(days=1)).date() }, "
            f"received {forecast_date.date()}."
        )

    history_age_days = int((today_utc - latest_date.normalize()).days)
    if history_age_days > MAX_HISTORY_AGE_DAYS and not args.allow_stale_history:
        raise RuntimeError(
            f"Latest observed demand date is {latest_date.date()} "
            f"({history_age_days} days old). Refresh the electricity dataset "
            "before operational use. For a historical/demo run only, add "
            "--allow-stale-history."
        )

    thresholds = load_thresholds()

    logger.info("Latest observed demand date: %s", latest_date.date())
    logger.info("Forecast date: %s", forecast_date.date())
    logger.info("History age: %d days", history_age_days)

    forecast_results = {}
    feature_metadata = {}
    for target in TARGETS:
        logger.info("Generating forecast for %s", target)
        forecast_results[target], feature_metadata[target] = forecast_target(
            history, forecast_date, target, thresholds
        )

    energy_category = forecast_results["energy_MU"]["demand_pressure_category"]
    peak_category = forecast_results["peak_demand_MW"]["demand_pressure_category"]
    primary_category = combine_demand_categories(
        energy_category, peak_category
    )

    now_utc = datetime.now(timezone.utc).isoformat()
    output = {
        "project": "GridGuard AI",
        "forecast_type": "one_day_ahead_daily_demand",
        "generated_at_utc": now_utc,
        "forecast_date": str(forecast_date.date()),
        "latest_observed_demand_date": str(latest_date.date()),
        "history_age_days": history_age_days,
        "data_status": (
            "stale_historical_demo"
            if history_age_days > MAX_HISTORY_AGE_DAYS
            else "recent_history"
        ),
        "forecasts": forecast_results,
        "risk_assessment": {
            "primary_demand_risk_category": primary_category,
            "energy_pressure_category": energy_category,
            "peak_pressure_category": peak_category,
            "weather_context_category": "not_assessed",
            "weather_context_status": (
                "No point-in-time future weather forecast snapshot has been "
                "connected to this inference module."
            ),
            "interpretation": (
                "Demand category is based on fixed training-period P80/P90/P95 "
                "percentiles. It is statistical demand pressure, not a physical "
                "grid-overload or reserve-margin assessment."
            ),
            "weather_escalation_applied": False,
        },
        "feature_diagnostics": feature_metadata,
        "limitations": [
            "The models predict one day ahead, not multiple days recursively.",
            "The model bundles were trained with historical weather predictors.",
            "For this first inference build, the latest historical weather feature "
            "row is carried forward; it is not a weather forecast.",
            "Weather context remains not_assessed until point-in-time forecast "
            "snapshots with fetched_at and valid_time are integrated.",
            "Forecast quality depends on refreshing observed electricity demand "
            "and rebuilding the fused feature dataset.",
            "P80/P90/P95 categories are statistical demand-pressure labels, "
            "not verified physical grid stress.",
        ],
    }

    stamp = forecast_date.strftime("%Y%m%d")
    output_json = OUTPUT_DIR / f"demand_forecast_{stamp}.json"
    output_csv = OUTPUT_DIR / f"demand_forecast_{stamp}.csv"
    latest_json = OUTPUT_DIR / "latest_demand_forecast.json"

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    flat = {
        "generated_at_utc": now_utc,
        "forecast_date": str(forecast_date.date()),
        "latest_observed_demand_date": str(latest_date.date()),
        "history_age_days": history_age_days,
        "data_status": output["data_status"],
        "energy_forecast_MU": forecast_results["energy_MU"]["forecast_value"],
        "energy_change_MU": forecast_results["energy_MU"]["predicted_change"],
        "energy_pressure": energy_category,
        "peak_forecast_MW": forecast_results["peak_demand_MW"]["forecast_value"],
        "peak_change_MW": forecast_results["peak_demand_MW"]["predicted_change"],
        "peak_pressure": peak_category,
        "primary_demand_risk": primary_category,
        "weather_context": "not_assessed",
        "weather_escalation_applied": False,
    }
    pd.DataFrame([flat]).to_csv(output_csv, index=False)
    with latest_json.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    logger.info("Forecast results:")
    logger.info(
        "Energy: %.3f MU | pressure=%s",
        forecast_results["energy_MU"]["forecast_value"],
        energy_category,
    )
    logger.info(
        "Peak: %.3f MW | pressure=%s",
        forecast_results["peak_demand_MW"]["forecast_value"],
        peak_category,
    )
    logger.info("Primary demand risk: %s", primary_category)
    logger.info("Weather context: not_assessed (no forecast snapshot connected)")
    logger.info("Saved JSON: %s", output_json)
    logger.info("Saved CSV: %s", output_csv)
    logger.info("Updated latest forecast: %s", latest_json)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Daily demand inference failed.")
        raise
