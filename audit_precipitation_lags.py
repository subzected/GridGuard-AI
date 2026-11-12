
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

IMD_FILE = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OM_FILE = ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"

imd = pd.read_csv(IMD_FILE)
om = pd.read_csv(OM_FILE)

# Standardize location identifiers and dates.
imd["location_id"] = imd["location_id"].str.lower().str.strip()
om["location_id"] = om["location_id"].str.lower().str.strip()

imd["date"] = pd.to_datetime(imd["valid_time"]).dt.normalize()
om["date"] = pd.to_datetime(om["time"]).dt.normalize()

imd = imd[["date", "location_id", "rain_mm"]].rename(
    columns={"rain_mm": "imd_rain"}
)

om = om[["date", "location_id", "precipitation_sum"]].rename(
    columns={"precipitation_sum": "om_rain"}
)

print("\nPRECIPITATION DATE-LAG AUDIT")
print("=" * 75)

for location in sorted(set(imd["location_id"]) & set(om["location_id"])):
    a = imd[imd["location_id"] == location].drop(columns="location_id")
    b = om[om["location_id"] == location].drop(columns="location_id")

    df = a.merge(b, on="date", how="inner").sort_values("date")
    df = df.set_index("date")

    results = []

    for lag in range(-2, 3):
        # Positive lag compares IMD day D with Open-Meteo day D+lag.
        comparison = pd.concat(
            [
                df["imd_rain"],
                df["om_rain"].shift(-lag)
            ],
            axis=1
        ).dropna()

        if len(comparison) < 100:
            continue

        mae = (
            comparison["imd_rain"] - comparison["om_rain"]
        ).abs().mean()

        corr = comparison["imd_rain"].corr(
            comparison["om_rain"]
        )

        results.append({
            "lag_days": lag,
            "MAE_mm": mae,
            "correlation": corr,
            "records": len(comparison)
        })

    result = pd.DataFrame(results).sort_values("MAE_mm")

    print(f"\nLOCATION: {location.upper()}")
    print(result.round(3).to_string(index=False))
