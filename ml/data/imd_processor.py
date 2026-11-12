# USED

from pathlib import Path
import pandas as pd


INPUT_PATH = Path("data/raw/imd_daily_weather_2010_2025.csv")
OUTPUT_PATH = Path("data/interim/imd_normalized.csv")


def process_imd(input_path=INPUT_PATH, output_path=OUTPUT_PATH):

    df = pd.read_csv(input_path)

    required = [
        "valid_time", "location_id", "location_name",
        "latitude", "longitude", "tmin", "tmax",
        "temperature_c", "rain_mm"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing IMD columns: {sorted(missing)}")

    df["snapshot_date"] = pd.to_datetime(
        df["valid_time"], errors="coerce"
    ).dt.normalize()

    df["location_id"] = (
        df["location_id"].astype("string").str.strip().str.lower()
    )

    numeric_columns = [
        "latitude", "longitude", "tmin", "tmax",
        "temperature_c", "rain_mm"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df[[
        "snapshot_date",
        "location_id",
        "location_name",
        "latitude",
        "longitude",
        "tmin",
        "tmax",
        "temperature_c",
        "rain_mm"
    ]].copy()

    result = result.rename(columns={
        "temperature_c": "imd_temp_mean_c",
        "tmin": "imd_temp_min_c",
        "tmax": "imd_temp_max_c",
        "rain_mm": "imd_precipitation_mm"
    })

    result = result.dropna(
        subset=["snapshot_date", "location_id"]
    )

    if result.duplicated(
        ["location_id", "snapshot_date"]
    ).any():
        raise ValueError("Duplicate IMD location-date records found")

    result = result.sort_values(
        ["location_id", "snapshot_date"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("IMD preprocessing completed")
    print("Records:", len(result))
    print("Locations:", result["location_id"].nunique())
    print("Output:", output_path)

    return result


if __name__ == "__main__":
    process_imd()
