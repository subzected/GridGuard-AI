
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

imd = pd.read_csv(
    ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
)

om = pd.read_csv(
    ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"
)

imd["location_id"] = imd["location_id"].str.lower().str.strip()
om["location_id"] = om["location_id"].str.lower().str.strip()

imd["date"] = pd.to_datetime(imd["valid_time"])
om["date"] = pd.to_datetime(om["time"])

imd = imd[["date", "location_id", "rain_mm"]].rename(
    columns={"rain_mm": "imd_rain"}
)

om = om[["date", "location_id", "precipitation_sum"]].rename(
    columns={"precipitation_sum": "om_rain"}
)

df = imd.merge(om, on=["date", "location_id"], how="inner")

thresholds = {
    "Wet day (>=1 mm)": 1,
    "Heavy rain (>=10 mm)": 10,
    "Very heavy rain (>=50 mm)": 50,
}

print("\nWET-DAY AGREEMENT")
print("=" * 75)

for label, threshold in thresholds.items():
    actual = df["imd_rain"] >= threshold
    predicted = df["om_rain"] >= threshold

    tp = int((actual & predicted).sum())
    tn = int((~actual & ~predicted).sum())
    fp = int((~actual & predicted).sum())
    fn = int((actual & ~predicted).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    accuracy = (tp + tn) / len(df)

    print(f"\n{label}")
    print(f"True positives : {tp}")
    print(f"True negatives : {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"Precision      : {precision:.3f}")
    print(f"Recall         : {recall:.3f}")
    print(f"Accuracy       : {accuracy:.3f}")

print("\nAGREEMENT BY LOCATION AT 1 MM")
print("=" * 75)

for location, group in df.groupby("location_id"):
    actual = group["imd_rain"] >= 1
    predicted = group["om_rain"] >= 1

    tp = (actual & predicted).sum()
    fp = (~actual & predicted).sum()
    fn = (actual & ~predicted).sum()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"{location:12s} "
        f"precision={precision:.3f} "
        f"recall={recall:.3f}"
    )
