
from pathlib import Path
from datetime import datetime
import json

import pandas as pd

from ml.quality.spatial_validation import validate_spatial_data
from ml.quality.duplicate_detection import detect_duplicates
from ml.quality.source_comparison import compare_sources


ROOT = Path(__file__).resolve().parents[2]

IMD_PATH = ROOT / "data/raw/imd_daily_weather_2010_2025.csv"
OPENMETEO_PATH = ROOT / "data/raw/open_meteo/openmeteo_historical_all.csv"

OUTPUT_DIR = ROOT / "reports/weather_quality"


def inspect_dataset(path, date_column, location_column):
    """Run basic data integrity checks."""

    df = pd.read_csv(path)

    df[location_column] = df[location_column].astype(str).str.lower()

    dates = pd.to_datetime(df[date_column], errors="coerce")

    duplicate_keys = int(
        df.duplicated([location_column, date_column]).sum()
    )

    return {
        "file": str(path.relative_to(ROOT)),
        "records": int(len(df)),
        "locations": int(df[location_column].nunique()),
        "date_start": (
            str(dates.min().date()) if dates.notna().any() else None
        ),
        "date_end": (
            str(dates.max().date()) if dates.notna().any() else None
        ),
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_location_date_keys": duplicate_keys,
        "missing_values_by_column": {
            str(k): int(v)
            for k, v in df.isna().sum().items()
            if v > 0
        },
    }


def run_quality_control():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n======================================")
    print(" GRIDGUARD AI - WEATHER QUALITY CONTROL")
    print("======================================")

    print("\n[1/4] Inspecting raw datasets...")

    imd_integrity = inspect_dataset(
        IMD_PATH,
        "valid_time",
        "location_id"
    )

    om_integrity = inspect_dataset(
        OPENMETEO_PATH,
        "time",
        "location_id"
    )

    print("\n[2/4] Validating spatial data...")

    spatial = validate_spatial_data()

    print("\n[3/4] Detecting duplicate weather series...")

    duplicates = detect_duplicates()

    temp_duplicates = duplicates["temperature"]

    exact_temp_count = int(
        temp_duplicates["exact_duplicate_series"].sum()
    )

    near_temp_count = int(
        (
            temp_duplicates["near_duplicate_series"]
            & ~temp_duplicates["exact_duplicate_series"]
        ).sum()
    )

    print("\n[4/4] Comparing weather sources...")

    comparison = compare_sources()

    # --------------------------------------------------
    # Spatial validation summary
    # --------------------------------------------------

    # Count how many unique locations share each IMD grid.
    grid_counts = (
        spatial.dropna(subset=["weather_grid_id"])
        .groupby("weather_grid_id")["location_id"]
        .nunique()
    )

    # Count grid IDs associated with more than one location.
    shared_grid_count = int((grid_counts > 1).sum())

    # Count unique locations participating in shared grids.
    shared_grid_locations = int(
        grid_counts[grid_counts > 1].sum()
    )

    spatial_summary = {
        "locations_checked": int(len(spatial)),

        "good_locations": int(
            (spatial["spatial_quality"] == "GOOD").sum()
        ),

        "review_locations": int(
            (spatial["spatial_quality"] == "REVIEW").sum()
        ),

        "warning_locations": int(
            (spatial["spatial_quality"] == "WARNING").sum()
        ),

        "shared_imd_grid_groups": shared_grid_count,

        "locations_in_shared_imd_grids": shared_grid_locations,
    }

    # --------------------------------------------------
    # Source comparison summary
    # --------------------------------------------------

    comparison_summary = {
        "locations_compared": int(len(comparison)),

        "temperature_review_locations": (
            comparison.loc[
                comparison["temperature_quality"].isin(
                    ["REVIEW", "CRITICAL"]
                ),
                "location_id"
            ].tolist()
        ),

        "precipitation_review_locations": (
            comparison.loc[
                comparison["precipitation_quality"].isin(
                    ["REVIEW", "CRITICAL"]
                ),
                "location_id"
            ].tolist()
        ),
    }

    # --------------------------------------------------
    # Final quality report
    # --------------------------------------------------

    report = {
        "project": "GridGuard AI",
        "report_type": "Weather Quality Control",

        "generated_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "datasets": {
            "imd": imd_integrity,
            "openmeteo": om_integrity,
        },

        "spatial_validation": spatial_summary,

        "duplicate_detection": {
            "exact_temperature_duplicate_pairs": exact_temp_count,
            "near_duplicate_temperature_pairs": near_temp_count,
        },

        "source_comparison": comparison_summary,

        "important_notes": [
            "High correlation does not prove measurement agreement.",
            "Exact duplicate series are flagged, not automatically removed.",
            "IMD and Open-Meteo measurements remain source-specific.",
            "Quality thresholds are diagnostic and should be validated.",
            "Shared weather grids must not be counted as independent observations.",
        ],
    }

    output_path = OUTPUT_DIR / "quality_report.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, default=str)

    print("\n======================================")
    print(" QUALITY CONTROL COMPLETED")
    print("======================================")

    print("IMD records:", imd_integrity["records"])
    print("Open-Meteo records:", om_integrity["records"])

    print("Exact duplicate temperature pairs:", exact_temp_count)
    print("Near-duplicate temperature pairs:", near_temp_count)

    print("Locations compared:", comparison_summary["locations_compared"])

    print("Shared IMD grid groups:", shared_grid_count)
    print("Locations in shared IMD grids:", shared_grid_locations)

    print("\nReports saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_quality_control()