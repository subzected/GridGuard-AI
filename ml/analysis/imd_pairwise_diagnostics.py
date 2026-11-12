
import pandas as pd

INPUT = "data/raw/imd_daily_weather_2010_2025.csv"

PAIRS = [
    ("asansol", "purulia"),
    ("bankura", "bardhaman"),
    ("bankura", "durgapur"),
    ("haldia", "howrah"),
    ("darjeeling", "siliguri"),
]


def main():
    df = pd.read_csv(INPUT)

    df["location_id"] = df["location_id"].str.lower()

    pivot = df.pivot(
        index="valid_time",
        columns="location_id",
        values="temperature_c"
    )

    rows = []

    for a, b in PAIRS:
        difference = (pivot[a] - pivot[b]).abs()

        rows.append({
            "pair": f"{a}-{b}",
            "exact_days": (difference < 0.001).sum(),
            "within_0.1C": (difference < 0.1).sum(),
            "mean_abs_diff": difference.mean(),
            "max_abs_diff": difference.max(),
            "std_diff": (pivot[a] - pivot[b]).std()
        })

    result = pd.DataFrame(rows).round(3)

    print("\nPAIRWISE TEMPERATURE DIFFERENCES\n")
    print(result.to_string(index=False))

    result.to_csv(
        "reports/imd_pairwise_diagnostics.csv",
        index=False
    )

    print("\nSaved: reports/imd_pairwise_diagnostics.csv")


if __name__ == "__main__":
    main()
