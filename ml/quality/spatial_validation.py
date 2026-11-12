
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

IMD_PATH = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OPENMETEO_PATH = ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"
OPENMETEO_DIR = ROOT / "data/raw/open_meteo"

OUTPUT_DIR = ROOT / "reports/weather_quality"


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two geographic coordinates."""

    earth_radius_km = 6371.0

    lat1, lon1, lat2, lon2 = map(
        np.radians,
        [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius_km * np.arcsin(np.sqrt(a))


def load_elevation_metadata():
    """Read elevation values from individual Open-Meteo JSON files."""

    elevations = {}

    for path in OPENMETEO_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as file:
                metadata = json.load(file)

            location_id = path.stem.lower()

            if "elevation" in metadata:
                elevations[location_id] = metadata["elevation"]

        except (OSError, json.JSONDecodeError):
            continue

    return elevations


def validate_spatial_data():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    imd = pd.read_csv(IMD_PATH)
    om = pd.read_csv(OPENMETEO_PATH)

    imd["location_id"] = imd["location_id"].astype(str).str.lower()
    om["location_id"] = om["location_id"].astype(str).str.lower()

    # One metadata row per IMD location.
    imd_locations = (
        imd.groupby("location_id", as_index=False)
        .agg(
            location_name=("location_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            grid_latitude=("grid_latitude", "first"),
            grid_longitude=("grid_longitude", "first"),
        )
    )

    # One metadata row per Open-Meteo location.
    om_locations = (
        om.groupby("location_id", as_index=False)
        .agg(
            openmeteo_latitude=("latitude", "first"),
            openmeteo_longitude=("longitude", "first"),
            location_name_om=("location_name", "first"),
        )
    )

    merged = imd_locations.merge(
        om_locations,
        on="location_id",
        how="outer",
        indicator=True
    )

    elevations = load_elevation_metadata()
    rows = []

    for _, row in merged.iterrows():
        location_id = row["location_id"]

        result = {
            "location_id": location_id,
            "imd_location_latitude": row.get("latitude"),
            "imd_location_longitude": row.get("longitude"),
            "imd_grid_latitude": row.get("grid_latitude"),
            "imd_grid_longitude": row.get("grid_longitude"),
            "openmeteo_latitude": row.get("openmeteo_latitude"),
            "openmeteo_longitude": row.get("openmeteo_longitude"),
            "openmeteo_elevation_m": elevations.get(location_id),
            "source_match": row["_merge"],
        }

        if pd.notna(row.get("latitude")) and pd.notna(row.get("longitude")):
            result["location_to_imd_grid_km"] = haversine_km(
                row["latitude"],
                row["longitude"],
                row["grid_latitude"],
                row["grid_longitude"]
            )
        else:
            result["location_to_imd_grid_km"] = np.nan

        if (
            pd.notna(row.get("openmeteo_latitude"))
            and pd.notna(row.get("imd_grid_latitude"))
        ):
            result["openmeteo_to_imd_grid_km"] = haversine_km(
                row["openmeteo_latitude"],
                row["openmeteo_longitude"],
                row["imd_grid_latitude"],
                row["imd_grid_longitude"]
            )
        else:
            result["openmeteo_to_imd_grid_km"] = np.nan

        result["weather_grid_id"] = (
            f"{row['grid_latitude']:.2f}_{row['grid_longitude']:.2f}"
            if pd.notna(row.get("grid_latitude"))
            else None
        )

        distance = result["openmeteo_to_imd_grid_km"]

        if pd.isna(distance):
            result["spatial_quality"] = "NOT_COMPARABLE"
        elif distance <= 1:
            result["spatial_quality"] = "GOOD"
        elif distance <= 10:
            result["spatial_quality"] = "REVIEW"
        else:
            result["spatial_quality"] = "WARNING"

        rows.append(result)

    spatial = pd.DataFrame(rows)

    spatial.to_csv(
        OUTPUT_DIR / "spatial_validation.csv",
        index=False
    )

    # Detect multiple IMD location labels using the same grid cell.
    grid_groups = (
        imd_locations.groupby(
            ["grid_latitude", "grid_longitude"]
        )["location_id"]
        .agg(list)
        .reset_index()
    )

    grid_groups = grid_groups[
        grid_groups["location_id"].map(len) > 1
    ].copy()

    grid_groups["location_count"] = (
        grid_groups["location_id"].map(len)
    )

    grid_groups["location_ids"] = (
        grid_groups["location_id"].map(lambda x: ";".join(x))
    )

    grid_groups["weather_grid_id"] = grid_groups.apply(
        lambda row: (
            f"{row['grid_latitude']:.2f}_"
            f"{row['grid_longitude']:.2f}"
        ),
        axis=1
    )

    grid_groups.drop(columns=["location_id"]).to_csv(
        OUTPUT_DIR / "duplicate_grid_groups.csv",
        index=False
    )

    print("Spatial validation completed.")
    print("Locations checked:", len(spatial))
    print("Shared IMD grid groups:", len(grid_groups))

    return spatial


if __name__ == "__main__":
    validate_spatial_data()
