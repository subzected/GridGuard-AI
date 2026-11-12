
from pathlib import Path
from itertools import combinations

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

IMD_PATH = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OUTPUT_DIR = ROOT / "reports/weather_quality"


def compare_location_pairs(pivot, variable, tolerance=0.001):
    """Compare every pair of location time series."""

    results = []

    for location_a, location_b in combinations(pivot.columns, 2):

        a = pivot[location_a]
        b = pivot[location_b]

        valid = pd.concat([a, b], axis=1).dropna()

        if len(valid) == 0:
            continue

        difference = (valid.iloc[:, 0] - valid.iloc[:, 1]).abs()

        correlation = valid.iloc[:, 0].corr(valid.iloc[:, 1])

        exact_count = int((difference < tolerance).sum())

        results.append({
            "variable": variable,
            "location_a": location_a,
            "location_b": location_b,
            "valid_days": len(valid),
            "exact_days": exact_count,
            "near_duplicate_days": int((difference < 0.1).sum()),
            "exact_duplicate_series": exact_count == len(valid),
            "near_duplicate_series": bool((difference < 0.1).all()),
            "correlation": correlation,
            "mean_absolute_difference": difference.mean(),
            "max_absolute_difference": difference.max(),
        })

    return pd.DataFrame(results)


def detect_duplicates():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IMD_PATH)

    df["location_id"] = df["location_id"].astype(str).str.lower()

    temperature = df.pivot(
        index="valid_time",
        columns="location_id",
        values="temperature_c"
    )

    rainfall = df.pivot(
        index="valid_time",
        columns="location_id",
        values="rain_mm"
    )

    temp_results = compare_location_pairs(
        temperature,
        "temperature_c"
    )

    rain_results = compare_location_pairs(
        rainfall,
        "rain_mm"
    )

    temp_results.to_csv(
        OUTPUT_DIR / "temperature_pairwise_comparison.csv",
        index=False
    )

    rain_results.to_csv(
        OUTPUT_DIR / "rainfall_pairwise_comparison.csv",
        index=False
    )

    exact_temp = temp_results[
        temp_results["exact_duplicate_series"]
    ]

    near_temp = temp_results[
        temp_results["near_duplicate_series"]
        & ~temp_results["exact_duplicate_series"]
    ]

    exact_rain = rain_results[
        rain_results["exact_duplicate_series"]
    ]

    print("\n========== DUPLICATE DETECTION ==========")

    print("Exact temperature duplicate pairs:", len(exact_temp))
    print("Near-duplicate temperature pairs:", len(near_temp))
    print("Exact rainfall duplicate pairs:", len(exact_rain))

    print("\nExact temperature duplicates:")

    if not exact_temp.empty:
        print(
            exact_temp[
                ["location_a", "location_b", "valid_days"]
            ].to_string(index=False)
        )
    else:
        print("None found.")

    return {
        "temperature": temp_results,
        "rainfall": rain_results,
    }


if __name__ == "__main__":
    detect_duplicates()
