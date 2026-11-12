
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

IMD_PATH = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OPENMETEO_PATH = ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"

OUTPUT_DIR = ROOT / "reports/weather_quality"


def classify_temperature(mae, correlation):
    if pd.isna(mae) or pd.isna(correlation):
        return "INSUFFICIENT_DATA"

    if mae > 5:
        return "CRITICAL"

    if mae > 2 or correlation < 0.80:
        return "REVIEW"

    return "GOOD"


def classify_precipitation(mae, correlation):
    if pd.isna(mae) or pd.isna(correlation):
        return "INSUFFICIENT_DATA"

    if mae > 10:
        return "CRITICAL"

    if mae > 5 or correlation < 0.40:
        return "REVIEW"

    return "GOOD"


def calculate_metrics(group):
    imd_temp = group["imd_temp"]
    om_temp = group["om_temp"]

    imd_rain = group["imd_rain"]
    om_rain = group["om_rain"]

    temp_diff = imd_temp - om_temp
    rain_diff = imd_rain - om_rain

    return {
        "record_count": len(group),

        "temperature_bias_c": temp_diff.mean(),
        "temperature_mae_c": temp_diff.abs().mean(),
        "temperature_rmse_c": np.sqrt((temp_diff ** 2).mean()),
        "temperature_correlation": imd_temp.corr(om_temp),

        "rainfall_bias_mm": rain_diff.mean(),
        "rainfall_mae_mm": rain_diff.abs().mean(),
        "rainfall_rmse_mm": np.sqrt((rain_diff ** 2).mean()),
        "rainfall_correlation": imd_rain.corr(om_rain),

        "imd_wet_days": int((imd_rain >= 1).sum()),
        "openmeteo_wet_days": int((om_rain >= 1).sum()),
    }


def compare_sources():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imd = pd.read_csv(IMD_PATH)
    om = pd.read_csv(OPENMETEO_PATH)

    imd["location_id"] = imd["location_id"].str.lower()
    om["location_id"] = om["location_id"].str.lower()

    imd["snapshot_date"] = pd.to_datetime(imd["valid_time"])
    om["snapshot_date"] = pd.to_datetime(om["time"])

    imd = imd.rename(columns={
        "temperature_c": "imd_temp",
        "rain_mm": "imd_rain",
    })

    om = om.rename(columns={
        "temperature_2m_mean": "om_temp",
        "precipitation_sum": "om_rain",
    })

    required_imd = [
        "location_id",
        "snapshot_date",
        "imd_temp",
        "imd_rain",
    ]

    required_om = [
        "location_id",
        "snapshot_date",
        "om_temp",
        "om_rain",
    ]

    imd = imd[required_imd]
    om = om[required_om]

    # Prevent accidental many-to-many joins.
    if imd.duplicated(["location_id", "snapshot_date"]).any():
        raise ValueError("Duplicate IMD location/date keys detected.")

    if om.duplicated(["location_id", "snapshot_date"]).any():
        raise ValueError("Duplicate Open-Meteo location/date keys detected.")

    merged = imd.merge(
        om,
        on=["location_id", "snapshot_date"],
        how="inner"
    )

    merged = merged.dropna(
        subset=["imd_temp", "om_temp", "imd_rain", "om_rain"]
    )

    rows = []

    for location_id, group in merged.groupby("location_id"):

        metrics = calculate_metrics(group)

        metrics["location_id"] = location_id

        metrics["temperature_quality"] = classify_temperature(
            metrics["temperature_mae_c"],
            metrics["temperature_correlation"]
        )

        metrics["precipitation_quality"] = classify_precipitation(
            metrics["rainfall_mae_mm"],
            metrics["rainfall_correlation"]
        )

        rows.append(metrics)

    report = pd.DataFrame(rows)

    report.to_csv(
        OUTPUT_DIR / "source_comparison.csv",
        index=False
    )

    # Save daily comparisons for further investigation.
    merged.to_csv(
        OUTPUT_DIR / "daily_source_comparison.csv",
        index=False
    )

    print("\n========== SOURCE COMPARISON ==========")

    columns = [
        "location_id",
        "temperature_mae_c",
        "temperature_correlation",
        "rainfall_mae_mm",
        "rainfall_correlation",
        "temperature_quality",
        "precipitation_quality",
    ]

    print(
        report[columns]
        .round(3)
        .to_string(index=False)
    )

    return report


if __name__ == "__main__":
    compare_sources()
