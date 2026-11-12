from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
REPORT_DIR = ROOT / "reports" / "electricity" / "weather_ablation"
MODEL_DIR = ROOT / "ml" / "models" / "weather_ablation"
PREDICTION_DIR = ROOT / "data" / "processed" / "weather_ablation"

DATE_COL = "date"
TARGETS = ["energy_MU", "peak_demand_MW"]
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_STATE = 42

CALENDAR_FEATURES = [
    "day_of_week", "day_of_month", "month", "day_of_year",
    "week_of_year", "is_weekend", "day_of_year_sin", "day_of_year_cos",
]
INVALID_WEATHER_FEATURES = {
    "imd_locations_with_any_weather_feature",
    "open_meteo_locations_with_any_weather_feature",
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COL])
    required = {DATE_COL, *TARGETS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    if df[DATE_COL].duplicated().any():
        raise ValueError("Duplicate dates found in fused dataset.")
    for target in TARGETS:
        df[target] = pd.to_numeric(df[target], errors="coerce")
    return df.dropna(subset=TARGETS).reset_index(drop=True)


def split_data(df):
    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    val_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:val_end].copy(),
        df.iloc[val_end:].copy(),
    )


def build_feature_sets(df):
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    calendar = [c for c in CALENDAR_FEATURES if c in numeric]
    demand = [
        c for c in df.columns
        if c in numeric and (
            c.startswith("energy_MU_") or c.startswith("peak_demand_MW_")
        )
    ]
    weather = [
        c for c in df.columns
        if c in numeric
        and c.startswith(("imd_", "open_meteo_"))
        and c not in INVALID_WEATHER_FEATURES
    ]
    feature_sets = {
        "A_demand_history": calendar + demand,
        "B_weather_only": calendar + weather,
        "C_demand_plus_weather": calendar + demand + weather,
    }
    for name, columns in feature_sets.items():
        if not columns:
            raise ValueError(f"Feature set {name} is empty.")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate feature names in {name}.")
    return feature_sets


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("model", HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=20,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )),
    ])


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "R2": float(r2_score(actual, predicted)),
    }


def previous_day_baseline(validation, test, target):
    lag_col = f"{target}_lag_1d"
    result = {}
    for split_name, frame in [("validation", validation), ("test", test)]:
        prediction = pd.to_numeric(frame[lag_col], errors="coerce")
        valid = prediction.notna()
        result[split_name] = (
            {**metrics(frame.loc[valid, target], prediction.loc[valid]),
             "rows_evaluated": int(valid.sum())}
            if valid.any() else None
        )
    return result


def run_target(target, train, validation, test, feature_sets):
    target_report = {
        "target": target,
        "experiments": {},
        "previous_day_baseline": previous_day_baseline(validation, test, target),
    }
    prediction_frames = []

    for experiment_name, features in feature_sets.items():
        logging.info("%s | %s | %d features",
                     target, experiment_name, len(features))

        model = make_model()
        model.fit(
            train[features].replace([np.inf, -np.inf], np.nan),
            train[target].astype(float),
        )
        val_prediction = model.predict(
            validation[features].replace([np.inf, -np.inf], np.nan)
        )
        val_metrics = metrics(validation[target], val_prediction)

        # Refit on train + validation; test remains untouched.
        combined = pd.concat([train, validation], ignore_index=True)
        final_model = make_model()
        final_model.fit(
            combined[features].replace([np.inf, -np.inf], np.nan),
            combined[target].astype(float),
        )
        test_prediction = final_model.predict(
            test[features].replace([np.inf, -np.inf], np.nan)
        )
        test_metrics = metrics(test[target], test_prediction)

        model_path = MODEL_DIR / f"{target}_{experiment_name}.joblib"
        joblib.dump(final_model, model_path)

        target_report["experiments"][experiment_name] = {
            "feature_count": len(features),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "model_path": str(model_path),
            "features": features,
        }
        logging.info(
            "%s | %s | validation MAE=%.3f | test MAE=%.3f",
            target, experiment_name, val_metrics["MAE"], test_metrics["MAE"],
        )

        prediction_frames.append(pd.DataFrame({
            DATE_COL: test[DATE_COL].to_numpy(),
            "target": target,
            "experiment": experiment_name,
            "actual": test[target].to_numpy(),
            "predicted": test_prediction,
            "absolute_error": np.abs(test[target].to_numpy() - test_prediction),
        }))

    ranked = sorted(
        target_report["experiments"].items(),
        key=lambda item: item[1]["validation_metrics"]["MAE"],
    )
    target_report["best_experiment_by_validation_mae"] = ranked[0][0]
    return target_report, prediction_frames


def main():
    setup_logging()
    for folder in [REPORT_DIR, MODEL_DIR, PREDICTION_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, validation, test = split_data(df)
    feature_sets = build_feature_sets(df)

    logging.info("Valid observations: %d", len(df))
    logging.info("Train: %d | Validation: %d | Test: %d",
                 len(train), len(validation), len(test))
    logging.info("Train dates: %s to %s",
                 train[DATE_COL].min().date(), train[DATE_COL].max().date())
    logging.info("Validation dates: %s to %s",
                 validation[DATE_COL].min().date(), validation[DATE_COL].max().date())
    logging.info("Test dates: %s to %s",
                 test[DATE_COL].min().date(), test[DATE_COL].max().date())
    for name, cols in feature_sets.items():
        logging.info("Feature set %s: %d columns", name, len(cols))

    report = {
        "project": "GridGuard AI",
        "dataset": str(DATA_PATH),
        "method": "chronological weather ablation using HistGradientBoosting",
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_start": str(train[DATE_COL].min().date()),
            "train_end": str(train[DATE_COL].max().date()),
            "validation_start": str(validation[DATE_COL].min().date()),
            "validation_end": str(validation[DATE_COL].max().date()),
            "test_start": str(test[DATE_COL].min().date()),
            "test_end": str(test[DATE_COL].max().date()),
        },
        "feature_sets": feature_sets,
        "targets": {},
        "notes": [
            "Same valid target dates and chronological split across experiments.",
            "Same HistGradientBoosting hyperparameters across experiments.",
            "Imputation is fitted on training data only.",
            "Final model is refitted on train plus validation.",
            "Test data is not used to select feature configuration.",
            "Previous-day naive baseline is reported separately.",
            "Weather features are historical prior-day features, not same-day forecasts.",
            "Incorrectly named locations_with_any_weather_feature columns are excluded.",
            "Results are for daily West Bengal state-level demand.",
        ],
    }

    all_predictions = []
    for target in TARGETS:
        target_report, predictions = run_target(
            target, train, validation, test, feature_sets
        )
        report["targets"][target] = target_report
        all_predictions.extend(predictions)

    report_path = REPORT_DIR / "weather_ablation_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)

    predictions_path = PREDICTION_DIR / "weather_ablation_test_predictions.csv"
    pd.concat(all_predictions, ignore_index=True).to_csv(
        predictions_path, index=False
    )

    summary_rows = []
    for target, target_data in report["targets"].items():
        baseline = target_data["previous_day_baseline"]["test"]
        for experiment, result in target_data["experiments"].items():
            summary_rows.append({
                "target": target,
                "experiment": experiment,
                "feature_count": result["feature_count"],
                "validation_MAE": result["validation_metrics"]["MAE"],
                "test_MAE": result["test_metrics"]["MAE"],
                "test_RMSE": result["test_metrics"]["RMSE"],
                "test_R2": result["test_metrics"]["R2"],
                "baseline_test_MAE": baseline["MAE"] if baseline else None,
            })

    summary_path = REPORT_DIR / "weather_ablation_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    logging.info("Weather ablation completed.")
    logging.info("JSON report: %s", report_path)
    logging.info("Summary CSV: %s", summary_path)
    logging.info("Test predictions: %s", predictions_path)
    logging.info("Models: %s", MODEL_DIR)


if __name__ == "__main__":
    main()
