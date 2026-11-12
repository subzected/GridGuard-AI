"""
Focused heat-weather ablation for GridGuard AI.

Compares three feature groups using identical chronological splits:
  A. Calendar + demand-history features
  B. Calendar + demand-history + heat features
  C. Calendar + demand-history + all historical-weather features

This is a historical predictive-value experiment, not an operational weather-
forecast backtest. Historical weather features are prior-only rolling features.

Run from the project root:
    python -m ml.evaluation.heat_weather_ablation
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "west_bengal_demand_weather_dataset.csv"
OUTPUT_DIR = ROOT / "reports" / "electricity" / "heat_weather_ablation"
TABLE_DIR = OUTPUT_DIR / "tables"

DATE_COLUMN = "date"
TARGETS = {
    "energy_MU": "energy_MU",
    "peak_demand_MW": "peak_demand_MW",
}
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
RANDOM_STATE = 42

# These columns are diagnostics, not model predictors.
EXCLUDED_NAME_PATTERNS = (
    r"location.*count",
    r"count.*location",
    r"coverage",
    r"available.*location",
    r"missing.*location",
)


def _is_diagnostic(column: str) -> bool:
    name = column.lower()
    return any(re.search(pattern, name) for pattern in EXCLUDED_NAME_PATTERNS)

def _is_calendar_feature(column: str) -> bool:
    name = column.lower()

    exact_names = {
        "year",
        "month",
        "day",
        "dayofweek",
        "day_of_week",
        "dayofyear",
        "day_of_year",
        "week",
        "weekofyear",
        "week_of_year",
        "quarter",
        "is_weekend",
        "weekend",
    }

    if name in exact_names:
        return True

    # Cyclical calendar features, e.g. month_sin, day_of_week_cos
    calendar_prefixes = (
        "month_sin", "month_cos",
        "day_sin", "day_cos",
        "dayofweek_sin", "dayofweek_cos",
        "day_of_week_sin", "day_of_week_cos",
        "dayofyear_sin", "dayofyear_cos",
        "day_of_year_sin", "day_of_year_cos",
    )

    return name.startswith(calendar_prefixes)


def _is_demand_history_feature(column: str) -> bool:
    name = column.lower()
    demand_terms = (
        "lag_", "_lag", "rolling_", "_rolling", "roll_",
        "change", "delta", "diff_", "difference",
    )
    target_terms = ("energy", "peak", "demand", "mu", "mw")
    return any(term in name for term in demand_terms) and any(
        term in name for term in target_terms
    )


def _is_weather_feature(column: str) -> bool:
    name = column.lower()
    weather_terms = (
        "rain", "wet_day", "wet_days", "consecutive_wet",
        "temp", "temperature", "humidity", "wind", "pressure",
        "precip", "relative_humidity",
    )
    return any(term in name for term in weather_terms)


def _is_heat_feature(column: str) -> bool:
    name = column.lower()
    return ("temp" in name or "temperature" in name) and _is_weather_feature(column)


def build_feature_groups(columns: List[str]) -> Dict[str, List[str]]:
    usable = [
        c for c in columns
        if c not in TARGETS.values()
        and c != DATE_COLUMN
        and not _is_diagnostic(c)
        
    ]
    # The dtype check is performed against the loaded dataframe in main().
    calendar = [c for c in usable if _is_calendar_feature(c)]
    demand = [c for c in usable if _is_demand_history_feature(c)]
    weather = [c for c in usable if _is_weather_feature(c)]
    heat = [c for c in weather if _is_heat_feature(c)]

    base = sorted(set(calendar + demand))
    heat_group = sorted(set(base + heat))
    all_weather = sorted(set(base + weather))

    if not calendar:
        raise ValueError("No calendar features were detected. Check dataset column names.")
    if not demand:
        raise ValueError(
            "No demand-history features were detected. Expected lag/rolling/change "
            "columns containing energy/peak/demand target names."
        )
    if not heat:
        raise ValueError(
            "No temperature/heat features were detected. Inspect weather column names "
            "and update _is_heat_feature() if your schema uses different names."
        )
    if not weather:
        raise ValueError("No historical weather features were detected.")

    return {
        "A_demand_calendar": base,
        "B_demand_calendar_heat": heat_group,
        "C_demand_calendar_all_weather": all_weather,
    }


def make_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error": float(np.mean(y_pred - y_true)),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run the demand-weather dataset builder first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    missing_targets = [c for c in TARGETS.values() if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    # Only retain rows with both target values. Preserve chronological ordering.
    valid = df.dropna(subset=list(TARGETS.values())).copy()
    valid = valid.sort_values(DATE_COLUMN).reset_index(drop=True)

    n = len(valid)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = int(n * (TRAIN_FRACTION + VALIDATION_FRACTION))

    train = valid.iloc[:train_end]
    validation = valid.iloc[train_end:validation_end]
    test = valid.iloc[validation_end:]

    numeric_columns = set(
        valid.select_dtypes(include=[np.number]).columns.tolist()
    )
    candidate_columns = [
        c for c in valid.columns
        if c in numeric_columns and c not in TARGETS.values()
        and c != DATE_COLUMN and not _is_diagnostic(c)
    ]

    # Build groups only from numeric predictors, avoiding target leakage.
    groups = build_feature_groups(candidate_columns)

    print("=" * 72)
    print("GRIDGUARD AI — FOCUSED HEAT-WEATHER ABLATION")
    print("=" * 72)
    print(f"Valid observations : {n}")
    print(f"Train              : {len(train)} ({train[DATE_COLUMN].min().date()} to {train[DATE_COLUMN].max().date()})")
    print(f"Validation         : {len(validation)} ({validation[DATE_COLUMN].min().date()} to {validation[DATE_COLUMN].max().date()})")
    print(f"Test               : {len(test)} ({test[DATE_COLUMN].min().date()} to {test[DATE_COLUMN].max().date()})")
    for group_name, features in groups.items():
        print(f"{group_name}: {len(features)} features")

    results = []
    predictions = pd.DataFrame({DATE_COLUMN: test[DATE_COLUMN].values})

    for target_label, target_column in TARGETS.items():
        y_train = train[target_column].astype(float).to_numpy()
        y_validation = validation[target_column].astype(float).to_numpy()
        y_test = test[target_column].astype(float).to_numpy()

        for group_name, features in groups.items():
            # Select by validation MAE only; test is held back for final reporting.
            model = make_model()
            model.fit(train[features], y_train)
            validation_prediction = model.predict(validation[features])
            validation_metrics = regression_metrics(
                y_validation, validation_prediction
            )

            final_model = make_model()
            final_model.fit(
                pd.concat([train[features], validation[features]], axis=0),
                np.concatenate([y_train, y_validation]),
            )
            test_prediction = final_model.predict(test[features])
            test_metrics = regression_metrics(y_test, test_prediction)

            results.append(
                {
                    "target": target_label,
                    "feature_group": group_name,
                    "n_features": len(features),
                    "validation_mae": validation_metrics["mae"],
                    "validation_rmse": validation_metrics["rmse"],
                    "validation_r2": validation_metrics["r2"],
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                    "test_mean_error": test_metrics["mean_error"],
                }
            )
            predictions[f"{target_label}__{group_name}"] = test_prediction
            print(
                f"{target_label:15s} | {group_name:32s} | "
                f"val MAE={validation_metrics['mae']:.4f} | "
                f"test MAE={test_metrics['mae']:.4f} | "
                f"test RMSE={test_metrics['rmse']:.4f} | "
                f"test R2={test_metrics['r2']:.4f}"
            )

    results_df = pd.DataFrame(results)
    results_path = TABLE_DIR / "heat_weather_ablation_results.csv"
    predictions_path = TABLE_DIR / "heat_weather_ablation_test_predictions.csv"
    results_df.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    best_rows = {}
    for target in TARGETS:
        target_results = results_df[results_df["target"] == target]
        best = target_results.loc[target_results["validation_mae"].idxmin()]
        baseline = target_results[
            target_results["feature_group"] == "A_demand_calendar"
        ].iloc[0]
        best_rows[target] = {
            "selected_by_validation": str(best["feature_group"]),
            "validation_mae": float(best["validation_mae"]),
            "test_mae_of_validation_selected_group": float(best["test_mae"]),
            "baseline_test_mae": float(baseline["test_mae"]),
            "test_mae_change_vs_baseline_percent": float(
                (best["test_mae"] - baseline["test_mae"])
                / baseline["test_mae"] * 100.0
            ),
        }

    report = {
        "experiment": "focused_heat_weather_ablation",
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "target_definition": "Daily energy supplied (MU) and peak demand (MW)",
        "method": (
            "Chronological 70/15/15 split. HistGradientBoosting with median "
            "imputation. Feature group selected by validation MAE; test metrics "
            "reported once after refitting on train plus validation."
        ),
        "weather_caveat": (
            "Historical weather inputs are prior-only rolling features, not "
            "weather forecasts. Results do not establish operational forecast "
            "skill or causal weather effects."
        ),
        "split": {
            "train_rows": len(train),
            "validation_rows": len(validation),
            "test_rows": len(test),
            "train_start": str(train[DATE_COLUMN].min().date()),
            "train_end": str(train[DATE_COLUMN].max().date()),
            "validation_start": str(validation[DATE_COLUMN].min().date()),
            "validation_end": str(validation[DATE_COLUMN].max().date()),
            "test_start": str(test[DATE_COLUMN].min().date()),
            "test_end": str(test[DATE_COLUMN].max().date()),
        },
        "feature_counts": {k: len(v) for k, v in groups.items()},
        "selected_results": best_rows,
        "feature_names": groups,
    }
    report_path = OUTPUT_DIR / "heat_weather_ablation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(results_path)
    print(predictions_path)
    print(report_path)
    print("\nInterpretation:")
    print(
        "Compare B against A to assess the incremental value of heat features; "
        "compare C against A to assess all-weather features. Use validation "
        "metrics for model selection and test metrics for final generalisation."
    )


if __name__ == "__main__":
    main()
