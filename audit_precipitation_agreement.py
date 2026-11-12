
from pathlib import Path

import numpy as np
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

df["abs_error"] = (df["imd_rain"] - df["om_rain"]).abs()

df["rain_category"] = pd.cut(
    df["imd_rain"],
    bins=[-0.001, 0.1, 2.5, 15, 50, np.inf],
    labels=[
        "Dry",
        "Light",
        "Moderate",
        "Heavy",
        "Very heavy"
    ]
)

print("\nAGREEMENT BY IMD RAINFALL CATEGORY")
print("=" * 65)

print(
    df.groupby("rain_category", observed=False)
    .agg(
        records=("abs_error", "count"),
        mean_imd=("imd_rain", "mean"),
        mean_openmeteo=("om_rain", "mean"),
        mae=("abs_error", "mean"),
        median_error=("abs_error", "median")
    )
    .round(2)
    .to_string()
)

print("\nAGREEMENT BY LOCATION")
print("=" * 65)

print(
    df.groupby("location_id")
    .agg(
        records=("abs_error", "count"),
        mae=("abs_error", "mean"),
        correlation=("imd_rain", lambda x: x.corr(
            df.loc[x.index, "om_rain"]
        ))
    )
    .round(3)
    .to_string()
)
