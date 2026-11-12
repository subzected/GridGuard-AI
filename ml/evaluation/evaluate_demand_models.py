
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data/processed/west_bengal_demand_weather_dataset.csv"
PRED_PATH = ROOT / "data/processed/model_predictions/demand_test_predictions.csv"
OUTPUT_PATH = ROOT / "reports/electricity/model_training/monthly_test_errors.csv"


def calculate_metrics(actual, predicted):
    return {
        "MAE": round(mean_absolute_error(actual, predicted), 3),
        "RMSE": round(np.sqrt(mean_squared_error(actual, predicted)), 3),
        "R2": round(r2_score(actual, predicted), 4),
    }


def main():
    base = pd.read_csv(DATA_PATH, parse_dates=["date"])
    predictions = pd.read_csv(PRED_PATH, parse_dates=["date"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_results = []

    for target in ["energy_MU", "peak_demand_MW"]:

        print("\n" + "=" * 65)
        print(target)
        print("=" * 65)

        actual_col = f"{target}_actual"
        predicted_col = f"{target}_predicted"
        lag_col = f"{target}_lag_1d"

        # Keep only rows belonging to this target.
        pred = predictions[
            ["date", actual_col, predicted_col]
        ].dropna(
            subset=[actual_col, predicted_col]
        ).copy()

        if pred.empty:
            print("No valid predictions found.")
            continue

        # Merge previous-day demand for baseline comparison.
        pred = pred.merge(
            base[["date", lag_col]],
            on="date",
            how="left",
        )

        actual = pred[actual_col]
        selected = pred[predicted_col]

        print("\nSelected model:")
        print(calculate_metrics(actual, selected))

        # Evaluate baseline only where lag-1 is available.
        baseline = pred[lag_col]
        valid = baseline.notna()

        if valid.any():
            print("\nPrevious-day baseline:")
            print(
                calculate_metrics(
                    actual[valid],
                    baseline[valid],
                )
            )
        else:
            print("\nPrevious-day baseline unavailable.")

        # Monthly evaluation.
        pred["month"] = pred["date"].dt.to_period("M").astype(str)
        pred["absolute_error"] = (actual - selected).abs()

        monthly = pred.groupby("month").agg(
            observations=(actual_col, "count"),
            MAE=("absolute_error", "mean"),
        ).reset_index()

        monthly["target"] = target
        monthly_results.append(monthly)

        print("\nMonthly test MAE:")
        print(
            monthly.drop(columns="target")
            .round(2)
            .to_string(index=False)
        )

        print("\nWorst five months:")
        print(
            monthly.nlargest(5, "MAE")[
                ["month", "observations", "MAE"]
            ].round(2).to_string(index=False)
        )

    if monthly_results:
        result = pd.concat(monthly_results, ignore_index=True)
        result.to_csv(OUTPUT_PATH, index=False)

        print("\nEvaluation completed.")
        print("Saved monthly results:", OUTPUT_PATH)
    else:
        print("\nNo evaluation results generated.")


if __name__ == "__main__":
    main()
