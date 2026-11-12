# USED

from pathlib import Path
import pandas as pd


INPUT_PATH = Path(
    "data/raw/open_meteo/openmeteo_historical_all.csv"
)

OUTPUT_PATH = Path("data/interim/openmeteo_normalized.csv")


def process_openmeteo(
    input_path=INPUT_PATH,
    output_path=OUTPUT_PATH
):

    df = pd.read_csv(input_path)

    required = [
        "time", "location_id", "location_name",
        "latitude", "longitude",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "rain_sum",
        "wind_speed_10m_mean",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
        "wind_direction_10m_dominant",
        "pressure_msl_mean",
        "cloud_cover_mean",
        "weather_code"
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing Open-Meteo columns: {sorted(missing)}"
        )

    df["snapshot_date"] = pd.to_datetime(
        df["time"], errors="coerce"
    ).dt.normalize()

    df["location_id"] = (
        df["location_id"].astype("string").str.strip().str.lower()
    )

    numeric_columns = [
        c for c in required
        if c not in ["time", "location_id", "location_name"]
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    result = df.rename(columns={
        "temperature_2m_mean": "openmeteo_temp_mean_c",
        "temperature_2m_max": "openmeteo_temp_max_c",
        "temperature_2m_min": "openmeteo_temp_min_c",
        "relative_humidity_2m_mean": "humidity_pct",
        "precipitation_sum": "openmeteo_precipitation_mm",
        "rain_sum": "openmeteo_rain_mm",
        "wind_speed_10m_mean": "wind_speed_mean_kmh",
        "wind_speed_10m_max": "wind_speed_max_kmh",
        "wind_gusts_10m_max": "wind_gusts_max_kmh",
        "wind_direction_10m_dominant": "wind_direction_deg",
        "pressure_msl_mean": "pressure_hpa",
        "cloud_cover_mean": "cloud_cover_pct"
    })

    result = result[[
        "snapshot_date",
        "location_id",
        "location_name",
        "latitude",
        "longitude",
        "openmeteo_temp_mean_c",
        "openmeteo_temp_max_c",
        "openmeteo_temp_min_c",
        "humidity_pct",
        "openmeteo_precipitation_mm",
        "openmeteo_rain_mm",
        "wind_speed_mean_kmh",
        "wind_speed_max_kmh",
        "wind_gusts_max_kmh",
        "wind_direction_deg",
        "pressure_hpa",
        "cloud_cover_pct",
        "weather_code"
    ]].copy()

    result = result.dropna(
        subset=["snapshot_date", "location_id"]
    )

    if result.duplicated(
        ["location_id", "snapshot_date"]
    ).any():
        raise ValueError(
            "Duplicate Open-Meteo location-date records found"
        )

    result = result.sort_values(
        ["location_id", "snapshot_date"]
    ).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("Open-Meteo preprocessing completed")
    print("Records:", len(result))
    print("Locations:", result["location_id"].nunique())
    print("Output:", output_path)

    return result


if __name__ == "__main__":
    process_openmeteo()
