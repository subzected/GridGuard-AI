"""
Advanced ML training for West Bengal daily electricity demand forecasting.

Models:
    - XGBoost Regressor
    - LightGBM Regressor
    - Extra Trees Regressor

Targets:
    - energy_MU
    - peak_demand_MW

Protocol:
    - Chronological 70/15/15 train/validation/test split on valid target dates.
    - Validation MAE selects the model independently for each target.
    - Selected model is refitted on train + validation.
    - Test set is evaluated once.
    - Previous-day persistence (lag_1) is included as a baseline.
    - Test predictions are one-day-ahead: actual previous-day demand features
      are available when predicting each subsequent day.

Run from the project root:
    python -m ml.training.train_advanced_demand_models
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
INPUT_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
MODEL_DIR = ROOT / "ml" / "models" / "advanced_demand"
PREDICTION_DIR = ROOT / "data" / "processed" / "advanced_demand"
REPORT_DIR = ROOT / "reports" / "electricity" / "advanced_training"

TARGETS = ("energy_MU", "peak_demand_MW")
BAD_FEATURES = {
    "imd_locations_with_any_weather_feature",
    "open_meteo_locations_with_any_weather_feature",
}
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("advanced_demand_training")


def require_dependencies() -> None:
    missing = []
    if XGBRegressor is None:
        missing.append("xgboost")
    if LGBMRegressor is None:
        missing.append("lightgbm")

    if missing:
        package_list = " ".join(missing)
        raise ImportError(
            "Missing required training package(s): "
            f"{', '.join(missing)}. Install them in the active venv with:\n"
            f"python -m pip install {package_list}"
        )


def load_dataset() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}\n"
            "Run the demand dataset builder before training."
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    required = {"date", *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        duplicates = int(df["date"].duplicated().sum())
        raise ValueError(f"Found {duplicates} duplicate dates in the fused dataset.")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"date", *TARGETS, *BAD_FEATURES}
    columns = [
        column
        for column in df.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
    ]
    if not columns:
        raise ValueError("No numeric predictor columns were found.")
    return columns


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    valid = df.dropna(subset=list(TARGETS)).sort_values("date").reset_index(drop=True)
    n = len(valid)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = valid.iloc[:train_end].copy()
    validation = valid.iloc[train_end:validation_end].copy()
    test = valid.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Not enough valid target rows for a 70/15/15 split.")

    return train, validation, test


def build_candidates() -> dict[str, Any]:
    return {
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=600,
            max_features=0.85,
            min_samples_leaf=2,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=5,
            reg_lambda=5.0,
            objective="reg:squarederror",
            eval_metric="mae",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.85,
            reg_lambda=5.0,
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(model: Any) -> Pipeline:
    # Median imputation is learned only from the current fit partition.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", model),
        ]
    )


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def date_range(frame: pd.DataFrame) -> dict[str, str]:
    return {
        "start": frame["date"].min().strftime("%Y-%m-%d"),
        "end": frame["date"].max().strftime("%Y-%m-%d"),
    }


def main() -> None:
    require_dependencies()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    feature_columns = get_feature_columns(df)
    train, validation, test = chronological_split(df)

    logger.info("Dataset: %s", INPUT_PATH)
    logger.info("Rows: %d | Numeric features: %d", len(df), len(feature_columns))
    logger.info(
        "Split sizes: train=%d, validation=%d, test=%d",
        len(train), len(validation), len(test),
    )
    logger.info("Train dates: %s", date_range(train))
    logger.info("Validation dates: %s", date_range(validation))
    logger.info("Test dates: %s", date_range(test))

    candidates = build_candidates()
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT_PATH),
        "protocol": {
            "split": "chronological 70/15/15 over rows with both target values",
            "selection_metric": "validation MAE",
            "final_fit": "train + validation",
            "test_policy": "single final evaluation",
            "prediction_horizon": "one day ahead using prior observed demand lags",
            "imputation": "median, fitted separately inside each training pipeline",
        },
        "dataset": {
            "rows_total": int(len(df)),
            "rows_with_valid_targets": int(df[list(TARGETS)].notna().all(axis=1).sum()),
            "feature_count": len(feature_columns),
            "features": feature_columns,
            "train": {"rows": len(train), **date_range(train)},
            "validation": {"rows": len(validation), **date_range(validation)},
            "test": {"rows": len(test), **date_range(test)},
        },
        "targets": {},
    }

    all_predictions = pd.DataFrame({"date": test["date"].to_numpy()})
    summary_rows: list[dict[str, Any]] = []

    for target in TARGETS:
        logger.info("========== Training target: %s ==========", target)
        X_train = train[feature_columns]
        y_train = train[target].astype(float)
        X_val = validation[feature_columns]
        y_val = validation[target].astype(float)
        X_test = test[feature_columns]
        y_test = test[target].astype(float)

        validation_results: dict[str, dict[str, Any]] = {}
        best_name: str | None = None
        best_mae = float("inf")

        for model_name, estimator in candidates.items():
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_train, y_train)
            val_prediction = pipeline.predict(X_val)
            metrics = calculate_metrics(y_val.to_numpy(), val_prediction)

            validation_results[model_name] = {
                **metrics,
                "validation_rows": int(len(y_val)),
            }
            logger.info(
                "%s | validation MAE=%.4f RMSE=%.4f R2=%.4f",
                model_name, metrics["mae"], metrics["rmse"], metrics["r2"],
            )

            if metrics["mae"] < best_mae:
                best_mae = metrics["mae"]
                best_name = model_name

        assert best_name is not None
        logger.info("Selected %s for %s", best_name, target)

        # Rebuild a fresh estimator and fit on train + validation only.
        final_estimator = build_candidates()[best_name]
        final_pipeline = make_pipeline(final_estimator)
        fit_frame = pd.concat([train, validation], ignore_index=True)
        final_pipeline.fit(
            fit_frame[feature_columns],
            fit_frame[target].astype(float),
        )

        test_prediction = final_pipeline.predict(X_test)
        test_metrics = calculate_metrics(y_test.to_numpy(), test_prediction)

        # Persistence baseline is evaluated on exactly the same test rows
        # where the previous-day target is available.
        lag_column = f"{target}_lag_1"
        baseline_metrics = None
        if lag_column in test.columns:
            baseline_mask = test[lag_column].notna()
            if baseline_mask.any():
                baseline_metrics = calculate_metrics(
                    y_test.loc[baseline_mask].to_numpy(dtype=float),
                    test.loc[baseline_mask, lag_column].to_numpy(dtype=float),
                )

        model_path = MODEL_DIR / f"{target}_{best_name.lower()}.joblib"
        joblib.dump(
            {
                "pipeline": final_pipeline,
                "target": target,
                "model_name": best_name,
                "feature_columns": feature_columns,
                "selected_on": "validation_mae",
                "prediction_horizon": "one_day_ahead",
            },
            model_path,
        )

        prediction_column = f"{target}_predicted"
        all_predictions[f"{target}_actual"] = y_test.to_numpy()
        all_predictions[prediction_column] = test_prediction

        if lag_column in test.columns:
            all_predictions[f"{target}_persistence_lag1"] = test[lag_column].to_numpy()

        report["targets"][target] = {
            "selected_model": best_name,
            "validation_results": validation_results,
            "test_metrics": test_metrics,
            "persistence_baseline_test_metrics": baseline_metrics,
            "model_path": str(model_path),
            "test_rows": int(len(test)),
        }

        summary_rows.append({
            "target": target,
            "selected_model": best_name,
            "validation_mae": best_mae,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
            "persistence_test_mae": (
                baseline_metrics["mae"] if baseline_metrics else np.nan
            ),
        })

        logger.info(
            "%s | TEST MAE=%.4f RMSE=%.4f R2=%.4f",
            target, test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"],
        )
        if baseline_metrics:
            logger.info(
                "%s | persistence TEST MAE=%.4f",
                target, baseline_metrics["mae"],
            )

    predictions_path = PREDICTION_DIR / "advanced_demand_test_predictions.csv"
    summary_path = REPORT_DIR / "advanced_demand_model_summary.csv"
    report_path = REPORT_DIR / "advanced_demand_training_report.json"

    all_predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    report["outputs"] = {
        "predictions": str(predictions_path),
        "summary": str(summary_path),
        "report": str(report_path),
    }
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    logger.info("Training completed.")
    logger.info("Saved predictions: %s", predictions_path)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved report: %s", report_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Advanced demand model training failed.")
        sys.exit(1)
