
from pathlib import Path
import pandas as pd
import numpy as np

PATH = Path("data/processed/weather_merged.csv")

df = pd.read_csv(PATH, parse_dates=["snapshot_date"])

# -------------------------------------------------
# 1. Compare geographical coordinates
# -------------------------------------------------

print("\n========== COORDINATE AUDIT ==========")

imd = pd.read_csv("data/interim/imd_normalized.csv")
om = pd.read_csv("data/interim/openmeteo_normalized.csv")

coords = imd.groupby("location_id").agg(
    imd_latitude=("latitude", "first"),
    imd_longitude=("longitude", "first")
)

om_coords = om.groupby("location_id").agg(
    openmeteo_latitude=("latitude", "first"),
    openmeteo_longitude=("longitude", "first")
)

coords = coords.join(om_coords, how="left")

coords["latitude_difference"] = (
    coords["imd_latitude"] - coords["openmeteo_latitude"]
)

coords["longitude_difference"] = (
    coords["imd_longitude"] - coords["openmeteo_longitude"]
)

print(coords.round(5).to_string())

# -------------------------------------------------
# 2. Compare daily temperature and rainfall
# -------------------------------------------------

print("\n========== LOCATION STATISTICS ==========")

for location, group in df.groupby("location_id"):

    print(f"\n{location.upper()}")

    print(
        "IMD temperature mean:",
        round(group["imd_temp_mean_c"].mean(), 2)
    )

    print(
        "Open-Meteo temperature mean:",
        round(group["openmeteo_temp_mean_c"].mean(), 2)
    )

    print(
        "IMD rainfall total:",
        round(group["imd_precipitation_mm"].sum(), 2)
    )

    print(
        "Open-Meteo rainfall total:",
        round(group["openmeteo_precipitation_mm"].sum(), 2)
    )

    print(
        "IMD wet days:",
        (group["imd_precipitation_mm"] >= 1).sum()
    )

    print(
        "Open-Meteo wet days:",
        (group["openmeteo_precipitation_mm"] >= 1).sum()
    )

# -------------------------------------------------
# 3. Inspect Darjeeling monthly temperatures
# -------------------------------------------------

print("\n========== DARJEELING MONTHLY AUDIT ==========")

darjeeling = df[df["location_id"] == "darjeeling"].copy()

monthly = darjeeling.groupby(
    darjeeling["snapshot_date"].dt.month
).agg(
    imd_temp=("imd_temp_mean_c", "mean"),
    openmeteo_temp=("openmeteo_temp_mean_c", "mean"),
    imd_rain=("imd_precipitation_mm", "mean"),
    openmeteo_rain=("openmeteo_precipitation_mm", "mean")
)

monthly["temperature_bias"] = (
    monthly["imd_temp"] - monthly["openmeteo_temp"]
)

print(monthly.round(2).to_string())

# -------------------------------------------------
# 4. Check whether Kolkata and Howrah are identical
# -------------------------------------------------

print("\n========== KOLKATA / HOWRAH AUDIT ==========")

kolkata = df[df["location_id"] == "kolkata"].sort_values(
    "snapshot_date"
)

howrah = df[df["location_id"] == "howrah"].sort_values(
    "snapshot_date"
)

compare_columns = [
    "imd_temp_mean_c",
    "openmeteo_temp_mean_c",
    "imd_precipitation_mm",
    "openmeteo_precipitation_mm"
]

for col in compare_columns:
    identical = np.allclose(
        kolkata[col].to_numpy(),
        howrah[col].to_numpy(),
        equal_nan=True
    )

    print(f"{col}: identical = {identical}")

# -------------------------------------------------
# 5. Check extreme rainfall discrepancies
# -------------------------------------------------

print("\n========== LARGEST RAINFALL DIFFERENCES ==========")

df["rainfall_abs_difference"] = (
    df["imd_precipitation_mm"]
    - df["openmeteo_precipitation_mm"]
).abs()

extremes = df.nlargest(
    15, "rainfall_abs_difference"
)[[
    "snapshot_date",
    "location_id",
    "imd_precipitation_mm",
    "openmeteo_precipitation_mm",
    "rainfall_abs_difference"
]]

print(extremes.to_string(index=False))

print("\nDiagnostics completed.")
