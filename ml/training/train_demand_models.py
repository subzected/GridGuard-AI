"""
GridGuard AI - Daily electricity demand model training.

Run from project root:
    python -m ml.training.train_demand_models

Targets:
    energy_MU
    peak_demand_MW

The script uses chronological train/validation/test splits, compares
naive baselines, Random Forest, and HistGradientBoosting, selects
by validation MAE, and evaluates the selected model once on test data.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
MODEL_DIR = ROOT / "ml" / "models" / "demand"
REPORT_DIR = ROOT / "reports" / "electricity" / "model_training"
PREDICTION_DIR = ROOT / "data" / "processed" / "model_predictions"

TARGETS = ["energy_MU", "peak_demand_MW"]
DATE_COLUMN = "date"
EXCLUDED_FEATURES = {
    DATE_COLUMN,
    "energy_MU",
    "peak_demand_MW",
    # These two columns were identified as incorrectly named in the
    # initial fusion builder and are deliberately excluded.
    "imd_locations_with_any_weather_feature",
    "open_meteo_locations_with_any_weather_feature",
}

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_STATE = 42


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    required = {DATE_COLUMN, *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if df[DATE_COLUMN].duplicated().any():
        duplicates = int(df[DATE_COLUMN].duplicated().sum())
        raise ValueError(f"Found {duplicates} duplicate dates.")

    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")

    # Keep rows with both target values for a consistent chronological split.
    df = df.dropna(subset=TARGETS).copy()
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    if len(df) < 500:
        raise ValueError(f"Too few usable observations for training: {len(df)}")

    return df


def chronological_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = df.iloc[:train_end].copy()
    validation = df.iloc[train_end:validation_end].copy()
    test = df.iloc[validation_end:].copy()

    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("One of the chronological splits is empty.")

    return train, validation, test


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    features = [c for c in numeric_columns if c not in EXCLUDED_FEATURES]

    if not features:
        raise ValueError("No numeric predictor columns found.")

    # Prevent accidental non-finite values from entering sklearn.
    return features


def make_pipeline(model):
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", model),
        ]
    )


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denominator = np.maximum(np.abs(y_true), 1e-8)
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE_percent": mape,
    }


def baseline_prediction(frame: pd.DataFrame, target: str, lag: int) -> np.ndarray:
    column = f"{target}_lag_{lag}d"
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def evaluate_baselines(validation: pd.DataFrame, target: str) -> dict:
    results = {}
    for label, lag in [("naive_lag_1d", 1), ("seasonal_naive_lag_7d", 7)]:
        predictions = baseline_prediction(validation, target, lag)
        valid = np.isfinite(predictions)
        if valid.any():
            results[label] = regression_metrics(
                validation.loc[valid, target], predictions[valid]
            )
        else:
            results[label] = None
    return results


def train_target(
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[dict, pd.DataFrame]:
    logging.info("Training target: %s", target)

    X_train = train[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_train = train[target].astype(float)

    X_validation = validation[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_validation = validation[target].astype(float)

    X_test = test[feature_columns].replace([np.inf, -np.inf], np.nan)
    y_test = test[target].astype(float)

    candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=350,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
    }

    validation_results = {}
    baseline_results = evaluate_baselines(validation, target)

    best_name = None
    best_pipeline = None
    best_mae = float("inf")

    for name, estimator in candidates.items():
        pipeline = make_pipeline(estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_validation)
        metrics = regression_metrics(y_validation, predictions)
        validation_results[name] = metrics

        logging.info(
            "%s | %s validation MAE: %.4f",
            target,
            name,
            metrics["MAE"],
        )

        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            best_name = name
            best_pipeline = pipeline

    # Compare learned models with baselines on validation MAE.
    for baseline_name, metrics in baseline_results.items():
        if metrics is not None:
            logging.info(
                "%s | %s validation MAE: %.4f",
                target,
                baseline_name,
                metrics["MAE"],
            )
            if metrics["MAE"] < best_mae:
                best_mae = metrics["MAE"]
                best_name = baseline_name
                best_pipeline = None

    # Retrain the selected learned model on train + validation only.
    # The test set remains untouched until final evaluation.
    if best_name in candidates:
        combined = pd.concat([train, validation], ignore_index=True)
        final_pipeline = make_pipeline(candidates[best_name])
        final_pipeline.fit(
            combined[feature_columns].replace([np.inf, -np.inf], np.nan),
            combined[target].astype(float),
        )
        test_predictions = final_pipeline.predict(X_test)
        model_path = MODEL_DIR / f"{target}_{best_name}.joblib"
        joblib.dump(final_pipeline, model_path)
        selected_type = "trained_model"
    else:
        lag = 1 if best_name == "naive_lag_1d" else 7
        test_predictions = baseline_prediction(test, target, lag)
        model_path = None
        selected_type = "baseline"

    valid_test = np.isfinite(test_predictions)
    if not valid_test.any():
        raise ValueError(f"No valid test predictions for {target}.")

    test_metrics = regression_metrics(
        y_test.to_numpy()[valid_test],
        test_predictions[valid_test],
    )

    predictions_frame = pd.DataFrame(
        {
            DATE_COLUMN: test.loc[valid_test, DATE_COLUMN].to_numpy(),
            f"{target}_actual": y_test.to_numpy()[valid_test],
            f"{target}_predicted": test_predictions[valid_test],
            f"{target}_absolute_error": np.abs(
                y_test.to_numpy()[valid_test] - test_predictions[valid_test]
            ),
            "selected_model": best_name,
        }
    )

    report = {
        "target": target,
        "selected_model": best_name,
        "selected_model_type": selected_type,
        "model_path": str(model_path) if model_path else None,
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "feature_count": int(len(feature_columns)),
        "validation_metrics": validation_results,
        "validation_baselines": baseline_results,
        "test_metrics": test_metrics,
        "test_prediction_rows": int(valid_test.sum()),
    }

    logging.info(
        "%s | selected=%s | test MAE=%.4f | RMSE=%.4f | R2=%.4f",
        target,
        best_name,
        test_metrics["MAE"],
        test_metrics["RMSE"],
        test_metrics["R2"],
    )

    return report, predictions_frame


def main() -> None:
    setup_logging()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train, validation, test = chronological_split(df)
    features = get_feature_columns(df)

    logging.info("Loaded %d usable observations.", len(df))
    logging.info(
        "Train: %d (%s to %s)",
        len(train), train[DATE_COLUMN].min().date(), train[DATE_COLUMN].max().date()
    )
    logging.info(
        "Validation: %d (%s to %s)",
        len(validation), validation[DATE_COLUMN].min().date(), validation[DATE_COLUMN].max().date()
    )
    logging.info(
        "Test: %d (%s to %s)",
        len(test), test[DATE_COLUMN].min().date(), test[DATE_COLUMN].max().date()
    )
    logging.info("Predictor features: %d", len(features))

    all_reports = {
        "project": "GridGuard AI",
        "dataset": str(DATA_PATH),
        "split_method": "chronological",
        "split_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": 1.0 - TRAIN_FRACTION - VALIDATION_FRACTION,
        },
        "feature_columns": features,
        "targets": {},
        "notes": [
            "Rows with missing target values were excluded from supervised training.",
            "Median imputation is fitted on training data only; final selected model is refitted on train plus validation.",
            "Test data is not used for model selection.",
            "Weather predictors are historical prior-day features, not same-day weather forecasts.",
            "The two incorrectly named locations_with_any_weather_feature columns are excluded.",
            "Results measure daily state-level demand forecasting, not confirmed grid outages or feeder-level failures.",
        ],
    }

    prediction_frames = []

    for target in TARGETS:
        report, predictions = train_target(
            target, train, validation, test, features
        )
        all_reports["targets"][target] = report
        prediction_frames.append(predictions)

    predictions_output = PREDICTION_DIR / "demand_test_predictions.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        predictions_output, index=False
    )

    report_output = REPORT_DIR / "demand_model_comparison.json"
    with report_output.open("w", encoding="utf-8") as file:
        json.dump(all_reports, file, indent=2, default=str)

    logging.info("Training completed.")
    logging.info("Report: %s", report_output)
    logging.info("Test predictions: %s", predictions_output)
    logging.info("Saved models in: %s", MODEL_DIR)


if __name__ == "__main__":
    main()
