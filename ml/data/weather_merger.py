# USED

from pathlib import Path
import json
import pandas as pd


IMD_PATH = Path("data/interim/imd_normalized.csv")
OPENMETEO_PATH = Path("data/interim/openmeteo_normalized.csv")

OUTPUT_DIR = Path("data/processed")

MERGED_PATH = OUTPUT_DIR / "weather_merged.csv"
UNMATCHED_PATH = OUTPUT_DIR / "imd_unmatched_locations.csv"
REPORT_PATH = OUTPUT_DIR / "weather_merge_report.json"


def merge_weather():

    imd = pd.read_csv(IMD_PATH, parse_dates=["snapshot_date"])
    om = pd.read_csv(
        OPENMETEO_PATH, parse_dates=["snapshot_date"]
    )

    keys = ["location_id", "snapshot_date"]

    if imd.duplicated(keys).any():
        raise ValueError("IMD contains duplicate merge keys")

    if om.duplicated(keys).any():
        raise ValueError("Open-Meteo contains duplicate merge keys")

    imd_locations = set(imd["location_id"].dropna().unique())
    om_locations = set(om["location_id"].dropna().unique())

    matched_locations = sorted(imd_locations & om_locations)
    imd_only_locations = sorted(imd_locations - om_locations)
    om_only_locations = sorted(om_locations - imd_locations)

    # Retain only locations present in both sources.
    imd_matched = imd[
        imd["location_id"].isin(matched_locations)
    ].copy()

    om_matched = om[
        om["location_id"].isin(matched_locations)
    ].copy()

    merged = pd.merge(
        imd_matched,
        om_matched.drop(
            columns=["location_name", "latitude", "longitude"]
        ),
        on=keys,
        how="inner",
        validate="one_to_one"
    )

    # Source comparison features.
    merged["temperature_difference_c"] = (
        merged["imd_temp_mean_c"]
        - merged["openmeteo_temp_mean_c"]
    )

    merged["temperature_difference_abs_c"] = (
        merged["temperature_difference_c"].abs()
    )

    merged["precipitation_difference_mm"] = (
        merged["imd_precipitation_mm"]
        - merged["openmeteo_precipitation_mm"]
    )

    merged["precipitation_difference_abs_mm"] = (
        merged["precipitation_difference_mm"].abs()
    )

    merged["month"] = merged["snapshot_date"].dt.month
    merged["year"] = merged["snapshot_date"].dt.year
    merged["day_of_year"] = merged["snapshot_date"].dt.dayofyear

    merged = merged.sort_values(keys).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    merged.to_csv(MERGED_PATH, index=False)

    # Preserve IMD records without an Open-Meteo location match.
    unmatched = imd[
        imd["location_id"].isin(imd_only_locations)
    ].copy()

    unmatched.to_csv(UNMATCHED_PATH, index=False)

    # Comparison report.
    report = {
        "imd_records": int(len(imd)),
        "openmeteo_records": int(len(om)),
        "merged_records": int(len(merged)),
        "matched_locations": matched_locations,
        "imd_only_locations": imd_only_locations,
        "openmeteo_only_locations": om_only_locations,
        "matched_location_count": len(matched_locations),
        "unmatched_imd_records": int(len(unmatched)),
        "duplicate_merge_keys": int(
            merged.duplicated(keys).sum()
        ),
        "date_range": {
            "start": str(merged["snapshot_date"].min().date())
            if not merged.empty else None,
            "end": str(merged["snapshot_date"].max().date())
            if not merged.empty else None
        },
        "temperature_comparison": {
            "mean_absolute_difference_c": float(
                merged["temperature_difference_abs_c"].mean()
            ) if not merged.empty else None,
            "mean_difference_c": float(
                merged["temperature_difference_c"].mean()
            ) if not merged.empty else None
        },
        "precipitation_comparison": {
            "mean_absolute_difference_mm": float(
                merged["precipitation_difference_abs_mm"].mean()
            ) if not merged.empty else None
        }
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=4),
        encoding="utf-8"
    )

    print("\n========== WEATHER MERGE REPORT ==========")
    print(json.dumps(report, indent=4))

    print("\nSaved files:")
    print(MERGED_PATH)
    print(UNMATCHED_PATH)
    print(REPORT_PATH)

    return merged


if __name__ == "__main__":
    merge_weather()
