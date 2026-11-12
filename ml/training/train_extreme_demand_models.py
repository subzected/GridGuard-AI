"""
Extreme-demand model experiments for GridGuard AI.

Run from the project root:
    python -m ml.training.train_extreme_demand_models

Design:
- Uses the existing West Bengal daily demand + historical-weather dataset.
- Predicts next-day demand change (delta), then reconstructs demand level.
- Uses a chronological train/validation/test split.
- Defines the high-demand threshold from TRAIN actual demand only (80th percentile).
- Tests ordinary, demand-weighted, and upper-quantile delta models.
- Selects models using validation performance only.
- Keeps the final test set untouched until model selection is complete.
- Reports overall and high-demand performance against persistence baselines.

Important:
The high-demand evaluation subset is identified using actual demand, which is
valid for retrospective evaluation but is not available at prediction time.
The saved model bundle contains the training-derived threshold and feature list.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
MODEL_DIR = ROOT / "ml" / "models" / "extreme_demand"
PREDICTION_DIR = ROOT / "data" / "processed" / "extreme_demand"
REPORT_DIR = ROOT / "reports" / "electricity" / "extreme_demand"

TARGETS = ("energy_MU", "peak_demand_MW")
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
EXTREME_QUANTILE = 0.80
RANDOM_STATE = 42

LOG = logging.getLogger("extreme_demand_training")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dirs() -> None:
    for directory in (MODEL_DIR, PREDICTION_DIR, REPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run the demand-weather dataset builder first."
        )

    df = pd.read_csv(DATA_PATH)
    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for target in TARGETS:
        if target not in df.columns:
            raise ValueError(f"Missing target column: {target}")

    return df


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "energy_MU",
        "peak_demand_MW",
        "imd_locations_with_any_weather_feature",
        "openmeteo_locations_with_any_weather_feature",
    }

    features = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            features.append(column)

    if not features:
        raise ValueError("No numeric predictor columns found.")

    return features


def get_lag_column(df: pd.DataFrame, target: str, lag: int) -> str:
    candidates = [
        f"{target}_lag_{lag}d",
        f"{target}_lag{lag}",
        f"{target}_lag_{lag}",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Missing {lag}-day lag for {target}. Tried {candidates}."
    )


def prepare_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    features = select_feature_columns(df)
    lag_columns = {}

    for target in TARGETS:
        lag_columns[target] = get_lag_column(df, target, 1)

    required = list(dict.fromkeys([
        "date",
        *TARGETS,
        *features,
        *lag_columns.values(),
    ]))
    # Keep a shared date index for both targets, matching the previous
    # demand-change experiment.
    frame = df[required].copy()
    frame = frame.dropna(
        subset=[*TARGETS, *lag_columns.values()]
    ).sort_values("date").reset_index(drop=True)

    if len(frame) < 100:
        raise ValueError(
            f"Only {len(frame)} usable rows remain; insufficient for chronological split."
        )

    return frame, features, lag_columns


def chronological_split(frame: pd.DataFrame):
    n = len(frame)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()

    def date_summary(part: pd.DataFrame) -> dict[str, Any]:
        return {
            "rows": int(len(part)),
            "start": part["date"].min().date().isoformat(),
            "end": part["date"].max().date().isoformat(),
        }

    LOG.info(
        "Split: train=%d validation=%d test=%d",
        len(train), len(validation), len(test),
    )
    LOG.info("Train dates: %s", date_summary(train))
    LOG.info("Validation dates: %s", date_summary(validation))
    LOG.info("Test dates: %s", date_summary(test))

    return train, validation, test


def demand_weights(
    train_actual: pd.Series,
    quantile: float = EXTREME_QUANTILE,
) -> tuple[np.ndarray, float]:
    """Give more influence to higher-demand training observations."""
    threshold = float(train_actual.quantile(quantile))
    values = train_actual.to_numpy(dtype=float)

    # Smoothly increase weights above the training 80th percentile.
    # The cap prevents a few extreme rows from dominating the fit.
    weights = np.ones(len(values), dtype=float)
    above = values > threshold
    if above.any():
        scale = max(float(np.nanstd(values)), 1e-8)
        excess = np.maximum(values[above] - threshold, 0.0) / scale
        weights[above] = np.clip(1.0 + 2.0 * excess, 1.0, 4.0)

    return weights, threshold


def build_candidates(target: str) -> dict[str, Any]:
    candidates: dict[str, Any] = {
        "Ridge_delta": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=10.0)),
        ]),
        "ExtraTrees_weighted_delta": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesRegressor(
                n_estimators=450,
                min_samples_leaf=2,
                max_features=0.9,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "HistGradientBoosting_weighted_delta": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.04,
                max_leaf_nodes=15,
                l2_regularization=2.0,
                random_state=RANDOM_STATE,
            )),
        ]),
        "GradientBoosting_quantile_delta_q75": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                loss="quantile",
                alpha=0.75,
                n_estimators=250,
                learning_rate=0.035,
                max_depth=2,
                min_samples_leaf=8,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    if XGBRegressor is not None:
        candidates["XGBoost_weighted_delta"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.035,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="reg:squarederror",
                eval_metric="mae",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                tree_method="hist",
            )),
        ])
    else:
        LOG.warning("xgboost is not installed; skipping XGBoost candidate.")

    if LGBMRegressor is not None:
        candidates["LightGBM_weighted_delta"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=500,
                learning_rate=0.035,
                num_leaves=15,
                max_depth=-1,
                min_child_samples=25,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=5.0,
                objective="regression",
                verbosity=-1,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])
    else:
        LOG.warning("lightgbm is not installed; skipping LightGBM candidate.")

    return candidates


def fit_candidate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_delta_train: pd.Series,
    weights: np.ndarray,
    weighted: bool,
) -> None:
    if weighted:
        pipeline.fit(
            X_train,
            y_delta_train,
            model__sample_weight=weights,
        )
    else:
        pipeline.fit(X_train, y_delta_train)


def predict_level(
    pipeline: Pipeline,
    X: pd.DataFrame,
    lag1: pd.Series,
) -> np.ndarray:
    delta_prediction = np.asarray(pipeline.predict(X), dtype=float)
    return lag1.to_numpy(dtype=float) + delta_prediction


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, Any]:
    pair = pd.DataFrame({
        "actual": actual.to_numpy(dtype=float),
        "predicted": np.asarray(predicted, dtype=float),
    }).dropna()

    if pair.empty:
        return {"n": 0, "mae": None, "rmse": None, "r2": None, "mean_error": None}

    y = pair["actual"].to_numpy()
    p = pair["predicted"].to_numpy()
    return {
        "n": int(len(pair)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "r2": float(r2_score(y, p)) if len(pair) > 1 else None,
        "mean_error": float(np.mean(p - y)),
    }


def evaluate_predictions(
    actual: pd.Series,
    predicted: np.ndarray,
    extreme_mask: np.ndarray,
) -> dict[str, Any]:
    overall = metrics(actual, predicted)
    high_actual = actual.to_numpy(dtype=float)[extreme_mask]
    high_pred = np.asarray(predicted, dtype=float)[extreme_mask]

    return {
        "overall": overall,
        "high_demand": metrics(pd.Series(high_actual), high_pred),
    }


def baseline_predictions(
    part: pd.DataFrame,
    target: str,
    lag_columns: dict[str, str],
) -> dict[str, np.ndarray]:
    lag1 = part[lag_columns[target]].to_numpy(dtype=float)
    lag7_col = get_lag_column(part, target, 7)
    lag7 = part[lag7_col].to_numpy(dtype=float)

    return {
        "persistence": lag1,
        "trend_adjusted_persistence": lag1 + (lag1 - lag7) / 6.0,
    }


def model_is_weighted(name: str) -> bool:
    return "weighted" in name


def validation_selection_score(
    evaluation: dict[str, Any],
    persistence_evaluation: dict[str, Any],
) -> tuple[float, bool]:
    """
    Select using validation high-demand MAE, while preventing a model with
    severely degraded overall MAE from winning.

    Guardrail: overall MAE must be no more than 10% worse than persistence.
    A model need not beat persistence in the high-demand subset to be retained;
    its score will reveal that limitation in the report.
    """
    high_mae = evaluation["high_demand"]["mae"]
    overall_mae = evaluation["overall"]["mae"]
    persistence_overall = persistence_evaluation["overall"]["mae"]

    if high_mae is None or overall_mae is None or persistence_overall is None:
        return float("inf"), False

    allowed = overall_mae <= persistence_overall * 1.10
    if not allowed:
        return float("inf"), False

    # High-demand performance has priority; overall error is a small tie-breaker.
    score = high_mae + 0.10 * overall_mae
    return float(score), True


def fit_and_select_target(
    target: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    lag_columns: dict[str, str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    LOG.info("========== Extreme-demand target: %s ==========", target)

    lag1_col = lag_columns[target]
    train_actual = train[target].astype(float)
    extreme_threshold = float(train_actual.quantile(EXTREME_QUANTILE))

    train_weights, weight_threshold = demand_weights(train_actual)
    # The same train-derived threshold is used for validation and test.
    val_extreme_mask = validation[target].to_numpy(dtype=float) >= extreme_threshold
    test_extreme_mask = test[target].to_numpy(dtype=float) >= extreme_threshold

    X_train = train[features]
    X_val = validation[features]
    X_test = test[features]

    y_train_delta = (
        train[target].astype(float) - train[lag1_col].astype(float)
    )
    y_val = validation[target].astype(float)
    y_test = test[target].astype(float)

    val_baselines = baseline_predictions(validation, target, lag_columns)
    test_baselines = baseline_predictions(test, target, lag_columns)

    val_baseline_evals = {
        name: evaluate_predictions(y_val, prediction, val_extreme_mask)
        for name, prediction in val_baselines.items()
    }
    test_baseline_evals = {
        name: evaluate_predictions(y_test, prediction, test_extreme_mask)
        for name, prediction in test_baselines.items()
    }

    persistence_val = val_baseline_evals["persistence"]
    candidates = build_candidates(target)
    candidate_results = {}
    fitted_candidates = {}

    for name, pipeline in candidates.items():
        try:
            fit_candidate(
                pipeline,
                X_train,
                y_train_delta,
                train_weights,
                weighted=model_is_weighted(name),
            )
            val_prediction = predict_level(
                pipeline, X_val, validation[lag1_col]
            )
            evaluation = evaluate_predictions(
                y_val, val_prediction, val_extreme_mask
            )
            score, eligible = validation_selection_score(
                evaluation, persistence_val
            )
            candidate_results[name] = {
                "validation": evaluation,
                "selection_score": score if np.isfinite(score) else None,
                "eligible_under_overall_error_guardrail": eligible,
            }
            fitted_candidates[name] = pipeline

            LOG.info(
                "%s | val overall MAE=%.4f | val high-demand MAE=%.4f | eligible=%s",
                name,
                evaluation["overall"]["mae"],
                evaluation["high_demand"]["mae"],
                eligible,
            )
        except Exception:
            LOG.exception("Candidate failed: %s", name)

    eligible_names = [
        name for name, result in candidate_results.items()
        if result["eligible_under_overall_error_guardrail"]
    ]
    if not eligible_names:
        # Fallback: choose the lowest high-demand validation MAE, but mark
        # that no candidate met the overall-error guardrail.
        eligible_names = list(candidate_results)
        if not eligible_names:
            raise RuntimeError(f"All candidate models failed for {target}.")
        selected_name = min(
            eligible_names,
            key=lambda n: candidate_results[n]["validation"]["high_demand"]["mae"],
        )
        guardrail_met = False
    else:
        selected_name = min(
            eligible_names,
            key=lambda n: candidate_results[n]["selection_score"],
        )
        guardrail_met = True

    selected_pipeline = fitted_candidates[selected_name]
    LOG.info("Selected %s for %s (validation only)", selected_name, target)

    # Refit selected model on train + validation. The held-out test set is
    # not used for fitting, threshold selection, or candidate selection.
    train_validation = pd.concat([train, validation], ignore_index=True)
    X_train_validation = train_validation[features]
    y_train_validation_delta = (
        train_validation[target].astype(float)
        - train_validation[lag1_col].astype(float)
    )
    refit_weights, _ = demand_weights(
        train_validation[target].astype(float)
    )
    fit_candidate(
        selected_pipeline,
        X_train_validation,
        y_train_validation_delta,
        refit_weights,
        weighted=model_is_weighted(selected_name),
    )

    test_prediction = predict_level(
        selected_pipeline, X_test, test[lag1_col]
    )
    selected_test_eval = evaluate_predictions(
        y_test, test_prediction, test_extreme_mask
    )

    output = pd.DataFrame({
        "date": test["date"].to_numpy(),
        f"{target}_actual": y_test.to_numpy(),
        f"{target}_predicted": test_prediction,
        f"{target}_persistence": test_baselines["persistence"],
        f"{target}_trend_adjusted_persistence": (
            test_baselines["trend_adjusted_persistence"]
        ),
        f"{target}_is_high_demand": test_extreme_mask,
        f"{target}_train_p80_threshold": extreme_threshold,
    })

    bundle = {
        "pipeline": selected_pipeline,
        "target": target,
        "selected_model": selected_name,
        "feature_columns": features,
        "target_definition": "delta = target_t - target_lag_1d; level = lag_1d + predicted_delta",
        "forecast_horizon": "one day ahead",
        "extreme_quantile": EXTREME_QUANTILE,
        "extreme_threshold_train_only": extreme_threshold,
        "selection_criterion": (
            "Validation high-demand MAE + 0.10 * overall MAE; "
            "overall MAE constrained to <= 1.10 * validation persistence MAE"
        ),
        "overall_error_guardrail_met": guardrail_met,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "refit_rows": int(len(train_validation)),
        "test_rows": int(len(test)),
    }

    model_path = MODEL_DIR / f"{target}_{selected_name.lower()}_extreme_delta.joblib"
    joblib.dump(bundle, model_path)

    report = {
        "target": target,
        "selected_model": selected_name,
        "extreme_definition": {
            "quantile": EXTREME_QUANTILE,
            "threshold_source": "training actual demand only",
            "threshold": extreme_threshold,
            "validation_high_demand_rows": int(val_extreme_mask.sum()),
            "test_high_demand_rows": int(test_extreme_mask.sum()),
            "note": (
                "Actual demand defines retrospective high-demand evaluation groups. "
                "It is not available at forecast time."
            ),
        },
        "validation_baselines": val_baseline_evals,
        "validation_candidates": candidate_results,
        "test_baselines": test_baseline_evals,
        "test_selected_model": selected_test_eval,
        "model_bundle": str(model_path),
        "overall_error_guardrail_met": guardrail_met,
    }

    LOG.info(
        "%s TEST | overall MAE=%.4f | high-demand MAE=%.4f | "
        "persistence overall MAE=%.4f | persistence high-demand MAE=%.4f",
        target,
        selected_test_eval["overall"]["mae"],
        selected_test_eval["high_demand"]["mae"],
        test_baseline_evals["persistence"]["overall"]["mae"],
        test_baseline_evals["persistence"]["high_demand"]["mae"],
    )

    return report, output


def main() -> None:
    setup_logging()
    warnings.filterwarnings(
        "ignore",
        message=".*does not have valid feature names.*",
    )
    ensure_dirs()

    df = load_dataset()
    frame, features, lag_columns = prepare_frame(df)

    LOG.info("Loaded dataset: %s", DATA_PATH)
    LOG.info("Usable rows: %d | numeric features: %d", len(frame), len(features))
    LOG.info("Features include demand lags and prior-only historical weather.")

    train, validation, test = chronological_split(frame)

    reports = {
        "dataset": str(DATA_PATH),
        "target_definition": (
            "Predict next-day demand change and reconstruct demand level."
        ),
        "split": {
            "train": {
                "rows": len(train),
                "start": train["date"].min().date().isoformat(),
                "end": train["date"].max().date().isoformat(),
            },
            "validation": {
                "rows": len(validation),
                "start": validation["date"].min().date().isoformat(),
                "end": validation["date"].max().date().isoformat(),
            },
            "test": {
                "rows": len(test),
                "start": test["date"].min().date().isoformat(),
                "end": test["date"].max().date().isoformat(),
            },
        },
        "feature_count": len(features),
        "feature_columns": features,
        "extreme_quantile": EXTREME_QUANTILE,
        "targets": {},
        "caveats": [
            "High-demand thresholds are calculated from training actual demand only.",
            "Validation and test high-demand masks use actual demand for retrospective evaluation.",
            "Actual demand is not available at inference time to decide whether a day is extreme.",
            "Test metrics must not be used for further model selection or tuning.",
            "Weather predictors are historical prior-day observations, not archived operational forecasts.",
        ],
    }

    predictions = None
    for target in TARGETS:
        target_report, target_predictions = fit_and_select_target(
            target,
            train,
            validation,
            test,
            features,
            lag_columns,
        )
        reports["targets"][target] = target_report

        if predictions is None:
            predictions = target_predictions
        else:
            predictions = predictions.merge(
                target_predictions, on="date", how="inner"
            )

    if predictions is None or predictions.empty:
        raise RuntimeError("No predictions were generated.")

    prediction_path = (
        PREDICTION_DIR / "extreme_demand_test_predictions.csv"
    )
    predictions.to_csv(prediction_path, index=False)

    report_path = REPORT_DIR / "extreme_demand_training_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, default=str)

    summary_rows = []
    for target, result in reports["targets"].items():
        for split_name, evaluation in (
            ("validation", result["validation_baselines"]),
            ("test", result["test_baselines"]),
        ):
            for method, values in evaluation.items():
                summary_rows.append({
                    "target": target,
                    "split": split_name,
                    "method": method,
                    "overall_mae": values["overall"]["mae"],
                    "overall_rmse": values["overall"]["rmse"],
                    "high_demand_mae": values["high_demand"]["mae"],
                    "high_demand_rmse": values["high_demand"]["rmse"],
                })

        for split_name, evaluation in (
            ("validation_selected", result["validation_candidates"]),
        ):
            for method, values in evaluation.items():
                summary_rows.append({
                    "target": target,
                    "split": split_name,
                    "method": method,
                    "overall_mae": values["validation"]["overall"]["mae"],
                    "overall_rmse": values["validation"]["overall"]["rmse"],
                    "high_demand_mae": values["validation"]["high_demand"]["mae"],
                    "high_demand_rmse": values["validation"]["high_demand"]["rmse"],
                })

        selected_test = result["test_selected_model"]
        summary_rows.append({
            "target": target,
            "split": "test_selected_model",
            "method": result["selected_model"],
            "overall_mae": selected_test["overall"]["mae"],
            "overall_rmse": selected_test["overall"]["rmse"],
            "high_demand_mae": selected_test["high_demand"]["mae"],
            "high_demand_rmse": selected_test["high_demand"]["rmse"],
        })

    summary_path = REPORT_DIR / "extreme_demand_model_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    LOG.info("Extreme-demand training completed.")
    LOG.info("Predictions: %s", prediction_path)
    LOG.info("Summary: %s", summary_path)
    LOG.info("Report: %s", report_path)


if __name__ == "__main__":
    main()
