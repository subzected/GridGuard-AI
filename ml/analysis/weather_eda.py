
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_PATH = Path("data/processed/weather_merged.csv")
OUTPUT_DIR = Path("reports/weather_eda")
PLOT_DIR = OUTPUT_DIR / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():

    df = pd.read_csv(INPUT_PATH)

    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"], errors="coerce"
    )

    df["year"] = df["snapshot_date"].dt.year
    df["month"] = df["snapshot_date"].dt.month

    df["season"] = df["month"].map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon",
        9: "Monsoon", 10: "Post-Monsoon", 11: "Post-Monsoon"
    })

    return df


def annual_weather_analysis(df):

    annual = df.groupby(
        ["location_id", "year"]
    ).agg(
        imd_temperature=("imd_temp_mean_c", "mean"),
        openmeteo_temperature=("openmeteo_temp_mean_c", "mean"),
        imd_rainfall=("imd_precipitation_mm", "sum"),
        openmeteo_rainfall=("openmeteo_precipitation_mm", "sum")
    ).reset_index()

    annual.to_csv(
        OUTPUT_DIR / "annual_weather_statistics.csv",
        index=False
    )

    return annual


def seasonal_analysis(df):

    seasonal = df.groupby(
        ["location_id", "season"]
    ).agg(
        imd_temperature=("imd_temp_mean_c", "mean"),
        openmeteo_temperature=("openmeteo_temp_mean_c", "mean"),
        imd_rainfall=("imd_precipitation_mm", "mean"),
        openmeteo_rainfall=("openmeteo_precipitation_mm", "mean"),
        humidity=("humidity_pct", "mean"),
        wind_speed=("wind_speed_mean_kmh", "mean")
    ).reset_index()

    seasonal.to_csv(
        OUTPUT_DIR / "seasonal_weather_statistics.csv",
        index=False
    )

    return seasonal


def location_comparison(df):

    results = []

    for location, group in df.groupby("location_id"):

        temp_diff = (
            group["imd_temp_mean_c"]
            - group["openmeteo_temp_mean_c"]
        )

        rain_diff = (
            group["imd_precipitation_mm"]
            - group["openmeteo_precipitation_mm"]
        )

        results.append({
            "location_id": location,

            "temperature_bias_c": temp_diff.mean(),
            "temperature_mae_c": temp_diff.abs().mean(),
            "temperature_rmse_c": np.sqrt(
                np.mean(temp_diff ** 2)
            ),

            "rainfall_bias_mm": rain_diff.mean(),
            "rainfall_mae_mm": rain_diff.abs().mean(),
            "rainfall_rmse_mm": np.sqrt(
                np.mean(rain_diff ** 2)
            ),

            "temperature_correlation": group[
                ["imd_temp_mean_c", "openmeteo_temp_mean_c"]
            ].corr().iloc[0, 1],

            "rainfall_correlation": group[
                ["imd_precipitation_mm",
                 "openmeteo_precipitation_mm"]
            ].corr().iloc[0, 1],

            "imd_wet_days": int(
                (group["imd_precipitation_mm"] >= 1).sum()
            ),

            "openmeteo_wet_days": int(
                (group["openmeteo_precipitation_mm"] >= 1).sum()
            )
        })

    result = pd.DataFrame(results)

    result.to_csv(
        OUTPUT_DIR / "source_comparison_by_location.csv",
        index=False
    )

    return result


def monthly_analysis(df):

    monthly = df.groupby("month").agg(
        imd_temperature=("imd_temp_mean_c", "mean"),
        openmeteo_temperature=("openmeteo_temp_mean_c", "mean"),
        imd_rainfall=("imd_precipitation_mm", "mean"),
        openmeteo_rainfall=("openmeteo_precipitation_mm", "mean")
    ).reset_index()

    monthly.to_csv(
        OUTPUT_DIR / "monthly_weather_statistics.csv",
        index=False
    )

    return monthly


def plot_annual_temperature(annual):

    data = annual.groupby("year").agg(
        IMD=("imd_temperature", "mean"),
        OpenMeteo=("openmeteo_temperature", "mean")
    )

    plt.figure(figsize=(12, 5))

    plt.plot(data.index, data["IMD"], label="IMD")
    plt.plot(data.index, data["OpenMeteo"], label="Open-Meteo")

    plt.title("Annual Mean Temperature Comparison")
    plt.xlabel("Year")
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "annual_temperature_comparison.png",
        dpi=300
    )
    plt.close()


def plot_annual_rainfall(annual):

    data = annual.groupby("year").agg(
        IMD=("imd_rainfall", "mean"),
        OpenMeteo=("openmeteo_rainfall", "mean")
    )

    plt.figure(figsize=(12, 5))

    plt.plot(data.index, data["IMD"], label="IMD")
    plt.plot(data.index, data["OpenMeteo"], label="Open-Meteo")

    plt.title("Annual Rainfall Comparison")
    plt.xlabel("Year")
    plt.ylabel("Annual Rainfall (mm)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "annual_rainfall_comparison.png",
        dpi=300
    )
    plt.close()


def plot_monthly_temperature(monthly):

    plt.figure(figsize=(10, 5))

    plt.plot(
        monthly["month"],
        monthly["imd_temperature"],
        marker="o",
        label="IMD"
    )

    plt.plot(
        monthly["month"],
        monthly["openmeteo_temperature"],
        marker="o",
        label="Open-Meteo"
    )

    plt.xticks(range(1, 13))
    plt.xlabel("Month")
    plt.ylabel("Temperature (°C)")
    plt.title("Average Monthly Temperature")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "monthly_temperature.png",
        dpi=300
    )
    plt.close()


def plot_temperature_scatter(df):

    plt.figure(figsize=(7, 7))

    plt.scatter(
        df["imd_temp_mean_c"],
        df["openmeteo_temp_mean_c"],
        alpha=0.15
    )

    minimum = min(
        df["imd_temp_mean_c"].min(),
        df["openmeteo_temp_mean_c"].min()
    )

    maximum = max(
        df["imd_temp_mean_c"].max(),
        df["openmeteo_temp_mean_c"].max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("IMD Temperature (°C)")
    plt.ylabel("Open-Meteo Temperature (°C)")
    plt.title("IMD vs Open-Meteo Temperature")
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "temperature_scatter.png",
        dpi=300
    )
    plt.close()


def main():

    df = load_data()

    print("Loaded records:", len(df))
    print("Locations:", df["location_id"].nunique())

    annual = annual_weather_analysis(df)
    seasonal = seasonal_analysis(df)
    comparison = location_comparison(df)
    monthly = monthly_analysis(df)

    plot_annual_temperature(annual)
    plot_annual_rainfall(annual)
    plot_monthly_temperature(monthly)
    plot_temperature_scatter(df)

    print("\nLOCATION COMPARISON")
    print(comparison.round(3).to_string(index=False))

    print("\nSEASONAL STATISTICS")
    print(seasonal.round(2).to_string(index=False))

    print("\nEDA completed.")
    print("Reports:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
