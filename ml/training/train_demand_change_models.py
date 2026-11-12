"""
Trend-aware daily electricity demand-change modelling for West Bengal.

For each target:
    delta_t = demand_t - demand_(t-1)
    predicted_demand_t = demand_(t-1) + predicted_delta_t

Models:
    - Linear Regression
    - Extra Trees
    - XGBoost
    - LightGBM

Baselines:
    - Persistence: previous-day demand
    - Trend-adjusted persistence:
          lag_1d + (lag_1d - lag_7d) / 6

Protocol:
    - Chronological 70/15/15 split on dates with valid targets and lag-1 values.
    - Model selection uses reconstructed demand-level validation MAE.
    - Final model is refitted on train + validation.
    - Test set is evaluated once.
    - No model is selected using test performance.

Run from project root:
    python -m ml.training.train_demand_change_models
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT / "data" / "processed"
    / "west_bengal_demand_weather_dataset.csv"
)
MODEL_DIR = ROOT / "ml" / "models" / "demand_change"
PREDICTION_DIR = ROOT / "data" / "processed" / "demand_change"
REPORT_DIR = ROOT / "reports" / "electricity" / "demand_change"

TARGETS = ("energy_MU", "peak_demand_MW")
BAD_FEATURES = {
    "imd_locations_with_any_weather_feature",
    "open_meteo_locations_with_any_weather_feature",
}
RANDOM_STATE = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("demand_change_training")


def require_dataset() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Fused dataset not found: {INPUT_PATH}\n"
            "Run ml.data.demand_dataset_builder first."
        )


def load_dataset() -> pd.DataFrame:
    require_dataset()
    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = {"date", *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].isna().any():
        raise ValueError("The dataset contains invalid dates.")
    if df["date"].duplicated().any():
        raise ValueError("The dataset contains duplicate dates.")

    for target in TARGETS:
        lag_col = f"{target}_lag_1d"
        if lag_col not in df.columns:
            raise ValueError(
                f"Required prior-day feature '{lag_col}' is missing. "
                "Check demand feature engineering."
            )

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Use numeric predictors already present in the fused dataset.

    Current-day target columns are excluded. Lagged demand, rolling demand,
    calendar and historical weather features remain eligible.
    """
    excluded = {"date", *TARGETS, *BAD_FEATURES}
    features = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not features:
        raise ValueError("No numeric predictor columns were found.")

    # Explicitly prevent accidental inclusion of future/current target aliases.
    suspicious = [
        col for col in features
        if col.endswith(("_target", "_actual", "_future"))
    ]
    if suspicious:
        raise ValueError(
            "Potential target leakage columns found among predictors: "
            f"{suspicious}"
        )

    return features


def prepare_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Require both targets and their previous-day values so both target models
    use the same chronological dates and can be compared consistently.
    """
    required = list(TARGETS) + [f"{target}_lag_1d" for target in TARGETS]
    frame = df.dropna(subset=required).copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    if len(frame) < 100:
        raise ValueError(
            f"Only {len(frame)} rows remain after requiring target and lag-1 "
            "values. At least 100 are required."
        )

    for target in TARGETS:
        frame[f"{target}_delta"] = (
            frame[target].astype(float)
            - frame[f"{target}_lag_1d"].astype(float)
        )

    return frame


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(frame)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    return train, validation, test


def build_candidates() -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_features=0.85,
            min_samples_leaf=2,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBRegressor is not None:
        candidates["XGBoost"] = XGBRegressor(
            n_estimators=600,
            max_depth=3,
            learning_rate=0.025,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=8.0,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        logger.warning("XGBoost unavailable; skipping XGBoost candidate.")

    if LGBMRegressor is not None:
        candidates["LightGBM"] = LGBMRegressor(
            n_estimators=600,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=8.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    else:
        logger.warning("LightGBM unavailable; skipping LightGBM candidate.")

    return candidates


def make_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("model", estimator),
        ]
    )


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual = actual_array - predicted_array

    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "r2": float(r2_score(actual_array, predicted_array)),
        "mean_error_actual_minus_predicted": float(np.mean(residual)),
    }


def evaluate_baselines(frame: pd.DataFrame, target: str) -> dict[str, Any]:
    lag1_col = f"{target}_lag_1d"
    lag7_col = f"{target}_lag_7d"

    actual = frame[target].astype(float)
    persistence = frame[lag1_col].astype(float)

    results: dict[str, Any] = {
        "persistence": {
            **calculate_metrics(actual, persistence),
            "rows": int(len(frame)),
        }
    }

    if lag7_col in frame.columns:
        valid = frame[lag7_col].notna()
        if valid.any():
            lag1 = frame.loc[valid, lag1_col].astype(float)
            lag7 = frame.loc[valid, lag7_col].astype(float)
            trend_prediction = lag1 + (lag1 - lag7) / 6.0
            results["trend_adjusted_persistence"] = {
                **calculate_metrics(
                    frame.loc[valid, target].astype(float),
                    trend_prediction,
                ),
                "rows": int(valid.sum()),
            }

    return results


def evaluate_candidate(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> tuple[np.ndarray, dict[str, float]]:
    delta_col = f"{target}_delta"
    lag1_col = f"{target}_lag_1d"

    predicted_delta = pipeline.predict(frame[features])
    predicted_level = (
        frame[lag1_col].to_numpy(dtype=float)
        + np.asarray(predicted_delta, dtype=float)
    )
    level_metrics = calculate_metrics(
        frame[target].to_numpy(dtype=float),
        predicted_level,
    )
    return predicted_level, level_metrics


def describe_partition(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_dataset()
    features = get_feature_columns(raw)
    frame = prepare_modeling_frame(raw)
    train, validation, test = chronological_split(frame)

    logger.info("Loaded dataset: %s", INPUT_PATH)
    logger.info(
        "Usable rows: %d | numeric features: %d",
        len(frame), len(features),
    )
    logger.info(
        "Split: train=%d, validation=%d, test=%d",
        len(train), len(validation), len(test),
    )
    logger.info("Train dates: %s", describe_partition(train))
    logger.info("Validation dates: %s", describe_partition(validation))
    logger.info("Test dates: %s", describe_partition(test))

    candidates = build_candidates()
    if not candidates:
        raise RuntimeError("No candidate models are available.")

    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "one-day-ahead demand-change prediction",
        "target_definition": (
            "delta_t = demand_t - demand_(t-1); "
            "level_prediction_t = demand_(t-1) + predicted_delta_t"
        ),
        "protocol": {
            "split": "chronological 70/15/15",
            "split_basis": (
                "rows with both targets and both previous-day lag features available"
            ),
            "selection_metric": "reconstructed demand-level validation MAE",
            "final_fit": "train + validation",
            "test_policy": "single final evaluation after validation selection",
            "imputation": "median fitted inside each model pipeline",
            "forecast_horizon": "one day ahead",
        },
        "dataset": {
            "source": str(INPUT_PATH),
            "raw_rows": int(len(raw)),
            "usable_rows": int(len(frame)),
            "feature_count": len(features),
            "features": features,
            "train": describe_partition(train),
            "validation": describe_partition(validation),
            "test": describe_partition(test),
        },
        "targets": {},
    }

    predictions = pd.DataFrame({"date": test["date"].to_numpy()})
    summary_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        logger.info("========== Target: %s ==========", target)

        delta_col = f"{target}_delta"
        lag1_col = f"{target}_lag_1d"

        X_train = train[features]
        y_train_delta = train[delta_col].astype(float)
        X_validation = validation[features]
        y_validation_level = validation[target].astype(float)

        validation_results: dict[str, Any] = {}
        best_name: str | None = None
        best_mae = float("inf")

        for model_name, estimator in candidates.items():
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_train, y_train_delta)

            validation_prediction, validation_metrics = evaluate_candidate(
                pipeline,
                validation,
                features,
                target,
            )
            validation_results[model_name] = {
                **validation_metrics,
                "rows": int(len(validation)),
            }

            logger.info(
                "%s | validation level MAE=%.4f RMSE=%.4f R2=%.4f",
                model_name,
                validation_metrics["mae"],
                validation_metrics["rmse"],
                validation_metrics["r2"],
            )

            if validation_metrics["mae"] < best_mae:
                best_mae = validation_metrics["mae"]
                best_name = model_name

        if best_name is None:
            raise RuntimeError(f"No model selected for {target}.")

        validation_baselines = evaluate_baselines(validation, target)
        logger.info(
            "%s | validation persistence MAE=%.4f",
            target, validation_baselines["persistence"]["mae"],
        )
        if "trend_adjusted_persistence" in validation_baselines:
            logger.info(
                "%s | validation trend-adjusted MAE=%.4f",
                target,
                validation_baselines["trend_adjusted_persistence"]["mae"],
            )

        logger.info("Selected %s for %s", best_name, target)

        # Refit only on train + validation, never on test.
        fit_frame = pd.concat([train, validation], ignore_index=True)
        final_estimator = build_candidates()[best_name]
        final_pipeline = make_pipeline(final_estimator)
        final_pipeline.fit(
            fit_frame[features],
            fit_frame[delta_col].astype(float),
        )

        test_prediction, test_metrics = evaluate_candidate(
            final_pipeline,
            test,
            features,
            target,
        )
        test_baselines = evaluate_baselines(test, target)

        model_path = MODEL_DIR / f"{target}_{best_name.lower()}_delta.joblib"
        bundle = {
            "pipeline": final_pipeline,
            "target": target,
            "model_name": best_name,
            "feature_columns": features,
            "delta_definition": "target_t - target_lag_1d",
            "level_reconstruction": "target_lag_1d + predicted_delta",
            "forecast_horizon": "one_day_ahead",
            "selected_on": "reconstructed_validation_level_mae",
        }
        joblib.dump(bundle, model_path)

        predictions[f"{target}_actual"] = test[target].to_numpy(dtype=float)
        predictions[f"{target}_lag_1d"] = test[lag1_col].to_numpy(dtype=float)
        predictions[f"{target}_predicted_delta"] = (
            test_prediction - test[lag1_col].to_numpy(dtype=float)
        )
        predictions[f"{target}_predicted"] = test_prediction
        predictions[f"{target}_persistence"] = test[lag1_col].to_numpy(dtype=float)

        lag7_col = f"{target}_lag_7d"
        if lag7_col in test.columns:
            lag7_valid = test[lag7_col].notna().to_numpy()
            trend_prediction = np.full(len(test), np.nan, dtype=float)
            lag1_values = test[lag1_col].to_numpy(dtype=float)
            lag7_values = test[lag7_col].to_numpy(dtype=float)
            trend_prediction[lag7_valid] = (
                lag1_values[lag7_valid]
                + (lag1_values[lag7_valid] - lag7_values[lag7_valid]) / 6.0
            )
            predictions[f"{target}_trend_adjusted_persistence"] = trend_prediction

        report["targets"][target] = {
            "selected_model": best_name,
            "validation_model_results": validation_results,
            "validation_baselines": validation_baselines,
            "test_model_metrics": test_metrics,
            "test_baselines": test_baselines,
            "model_path": str(model_path),
        }

        summary_rows.append({
            "target": target,
            "selected_model": best_name,
            "validation_mae": best_mae,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "persistence_validation_mae": (
                validation_baselines["persistence"]["mae"]
            ),
            "persistence_test_mae": test_baselines["persistence"]["mae"],
            "trend_adjusted_validation_mae": (
                validation_baselines.get(
                    "trend_adjusted_persistence", {}
                ).get("mae", np.nan)
            ),
            "trend_adjusted_test_mae": (
                test_baselines.get(
                    "trend_adjusted_persistence", {}
                ).get("mae", np.nan)
            ),
        })

        logger.info(
            "%s | TEST model MAE=%.4f RMSE=%.4f R2=%.4f",
            target, test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )
        logger.info(
            "%s | TEST persistence MAE=%.4f",
            target, test_baselines["persistence"]["mae"],
        )
        if "trend_adjusted_persistence" in test_baselines:
            logger.info(
                "%s | TEST trend-adjusted MAE=%.4f",
                target, test_baselines["trend_adjusted_persistence"]["mae"],
            )

    prediction_path = PREDICTION_DIR / "demand_change_test_predictions.csv"
    summary_path = REPORT_DIR / "demand_change_summary.csv"
    report_path = REPORT_DIR / "demand_change_comparison.json"

    predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    report["outputs"] = {
        "predictions": str(prediction_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    logger.info("Demand-change training completed.")
    logger.info("Predictions: %s", prediction_path)
    logger.info("Summary: %s", summary_path)
    logger.info("Report: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Demand-change model training failed.")
        sys.exit(1)
