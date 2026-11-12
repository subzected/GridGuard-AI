import logging
from pathlib import Path

import imdlib as imd
import numpy as np
import pandas as pd

from src.database.models import MONITORING_LOCATIONS

START_YEAR = 2010
END_YEAR = 2025

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "imd"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

CACHE_DIR = RAW_DIR / "cache"

OUTPUT_FILE = OUTPUT_DIR / "imd_daily_weather_2010_2025.csv"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

imd.cache.set_dir(str(CACHE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

logger = logging.getLogger(__name__)

def download_imd_dataset(variable):

    logger.info(
        "Downloading IMD %s data: %s-%s",
        variable,
        START_YEAR,
        END_YEAR
    )

    try:
        dataset = imd.load(
            variable,
            START_YEAR,
            END_YEAR,
            progress=True
        )

        logger.info(
            "Successfully loaded IMD dataset: %s",
            variable
        )

        return dataset.get_xarray()

    except Exception:
        logger.exception(
            "Unable to download/load IMD dataset: %s",
            variable
        )
        raise


# --------------------------------------------------
# EXTRACT NEAREST GRID CELL
# --------------------------------------------------

def extract_location_data(
    dataset,
    location,
    variable,
    missing_values
):

    data_array = dataset[variable]

    point = data_array.sel(
        lat=location["latitude"],
        lon=location["longitude"],
        method="nearest"
    )

    frame = point.to_dataframe(
        name=variable
    ).reset_index()

    frame["valid_time"] = pd.to_datetime(
        frame["time"]
    )

    frame["location_id"] = location["location_id"]
    frame["location_name"] = location["name"]

    frame["latitude"] = location["latitude"]
    frame["longitude"] = location["longitude"]

    # Record the actual grid cell used.
    frame["grid_latitude"] = float(point["lat"].values)
    frame["grid_longitude"] = float(point["lon"].values)

    frame[variable] = pd.to_numeric(
        frame[variable],
        errors="coerce"
    )

    frame[variable] = frame[variable].replace(
        missing_values,
        np.nan
    )

    return frame[
        [
            "valid_time",
            "location_id",
            "location_name",
            "latitude",
            "longitude",
            "grid_latitude",
            "grid_longitude",
            variable
        ]
    ]


# --------------------------------------------------
# BUILD DAILY HISTORICAL CSV
# --------------------------------------------------

def collect_historical_weather():

    logger.info("Starting historical IMD collection")

    # IMD official gridded datasets.
    rainfall = download_imd_dataset("rain")
    minimum_temperature = download_imd_dataset("tmin")
    maximum_temperature = download_imd_dataset("tmax")

    all_locations = []

    for location in MONITORING_LOCATIONS:

        logger.info(
            "Processing %s",
            location["name"]
        )

        rain_frame = extract_location_data(
            rainfall,
            location,
            "rain",
            [-999.0]
        )

        tmin_frame = extract_location_data(
            minimum_temperature,
            location,
            "tmin",
            [99.9, -999.0]
        )

        tmax_frame = extract_location_data(
            maximum_temperature,
            location,
            "tmax",
            [99.9, -999.0]
        )

        # Merge using observation dates and location.
        merged = rain_frame.merge(
            tmin_frame[
                [
                    "valid_time",
                    "location_id",
                    "tmin"
                ]
            ],
            on=["valid_time", "location_id"],
            how="outer"
        )

        merged = merged.merge(
            tmax_frame[
                [
                    "valid_time",
                    "location_id",
                    "tmax"
                ]
            ],
            on=["valid_time", "location_id"],
            how="outer"
        )

        merged["source"] = "imd-gridded"
        merged["temporal_resolution"] = "daily"

        merged["rain_mm"] = merged["rain"]
        merged["precipitation_mm"] = merged["rain"]

        merged["temperature_c"] = (
            merged["tmin"] + merged["tmax"]
        ) / 2

        all_locations.append(merged)

        logger.info(
            "%s: %s daily records",
            location["name"],
            len(merged)
        )

    final_frame = pd.concat(
        all_locations,
        ignore_index=True
    )

    final_frame = final_frame.sort_values(
        ["location_id", "valid_time"]
    )

    final_frame = final_frame.drop_duplicates(
        subset=["location_id", "valid_time"]
    )

    # Do not invent missing values.
    final_frame.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d"
    )

    logger.info("Historical IMD collection completed")

    logger.info(
        "Locations processed: %s",
        final_frame["location_id"].nunique()
    )

    logger.info(
        "Total rows: %s",
        len(final_frame)
    )

    logger.info(
        "Date range: %s to %s",
        final_frame["valid_time"].min(),
        final_frame["valid_time"].max()
    )

    logger.info(
        "Output: %s",
        OUTPUT_FILE
    )

    logger.info(
        "Missing rainfall values: %s",
        final_frame["rain_mm"].isna().sum()
    )

    return final_frame


if __name__ == "__main__":
    collect_historical_weather()
