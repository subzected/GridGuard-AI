
from pathlib import Path
import json
import numpy as np
import pandas as pd

IMD_PATH = Path("data/raw/imd_daily_weather_2010_2025.csv")
OM_PATH = Path("data/raw/open_meteo/openmeteo_historical_all.csv")
OM_DIR = Path("data/raw/open_meteo")


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0

    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def main():

    imd = pd.read_csv(IMD_PATH)
    om = pd.read_csv(OM_PATH)

    # -----------------------------------------
    # 1. IMD station and gridded coordinates
    # -----------------------------------------

    print("\n========== IMD GRID COORDINATES ==========")

    imd_locations = imd.groupby("location_id").agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        grid_latitude=("grid_latitude", "first"),
        grid_longitude=("grid_longitude", "first"),
        temperature=("temperature_c", "mean"),
        rainfall=("rain_mm", "mean")
    )

    print(imd_locations.round(4).to_string())

    # -----------------------------------------
    # 2. Open-Meteo returned coordinates
    # -----------------------------------------

    print("\n========== OPEN-METEO COORDINATES ==========")

    om_locations = om.groupby("location_id").agg(
        location_name=("location_name", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        temperature=("temperature_2m_mean", "mean"),
        rainfall=("precipitation_sum", "mean")
    )

    print(om_locations.round(4).to_string())

    # -----------------------------------------
    # 3. API response elevation
    # -----------------------------------------

    print("\n========== OPEN-METEO API ELEVATION ==========")

    for location in om["location_id"].unique():

        path = OM_DIR / f"{location.upper()}.json"

        if not path.exists():
            print(location, "JSON not found")
            continue

        payload = json.loads(path.read_text(encoding="utf-8"))

        print(
            location,
            "requested/returned latitude:",
            payload.get("latitude"),
            "longitude:",
            payload.get("longitude"),
            "elevation:",
            payload.get("elevation")
        )



    # -----------------------------------------
    # 4. Compare IMD grid and Open-Meteo points
    # -----------------------------------------

    print("\n========== SPATIAL DISTANCES ==========")

    # Normalise identifiers before joining.
    imd_locations.index = (
        imd_locations.index.astype(str).str.strip().str.lower()
    )

    om_locations.index = (
        om_locations.index.astype(str).str.strip().str.lower()
    )

    matched = imd_locations.join(
        om_locations,
        how="inner",
        lsuffix="_imd",
        rsuffix="_om"
    )

    for location, row in matched.iterrows():

        station_distance = haversine_km(
            row["latitude_imd"],
            row["longitude_imd"],
            row["latitude_om"],
            row["longitude_om"]
        )

        grid_distance = haversine_km(
            row["grid_latitude"],
            row["grid_longitude"],
            row["latitude_om"],
            row["longitude_om"]
        )

        print(
            f"{location}: "
            f"station distance={station_distance:.2f} km, "
            f"IMD-grid distance={grid_distance:.2f} km"
        )


    # -----------------------------------------
    # 5. Check duplicated IMD grid cells
    # -----------------------------------------

    print("\n========== DUPLICATED IMD GRID CELLS ==========")

    cells = imd_locations.reset_index().groupby(
        ["grid_latitude", "grid_longitude"]
    )["location_id"].apply(list)

    for cell, locations in cells.items():
        if len(locations) > 1:
            print(cell, "->", locations)

    # -----------------------------------------
    # 6. Compare Kolkata and Howrah
    # -----------------------------------------

    print("\n========== KOLKATA / HOWRAH GRID AUDIT ==========")

    for location in ["kolkata", "howrah"]:
        print(
            location,
            imd_locations.loc[
                location,
                ["grid_latitude", "grid_longitude"]
            ].to_dict()
        )

    # -----------------------------------------
    # 7. Inspect precipitation variable equality
    # -----------------------------------------

    print("\n========== OPEN-METEO RAINFALL VARIABLES ==========")

    print(
        "Max absolute difference between precipitation_sum "
        "and rain_sum:",
        (om["precipitation_sum"] - om["rain_sum"]).abs().max()
    )

    print("\nSpatial audit completed.")


if __name__ == "__main__":
    main()
