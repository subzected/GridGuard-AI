"""Exact feeder-name mapping to GridGuard location IDs."""

from __future__ import annotations

import re
import pandas as pd


MAPPING_REQUIRED = ["feeder_name", "location_id"]


def normalize_mapping_key(value):
    if pd.isna(value):
        return ""
    value = re.sub(r"\s+", " ", str(value)).strip().upper()
    return value


def map_feeders(
    events: pd.DataFrame,
    mapping_df: pd.DataFrame,
):
    """
    Map feeder names using exact normalized keys only.

    Returns mapped_df and unmapped_df. No fuzzy matching is performed because
    incorrectly merging electrical feeders can create false training labels.
    """
    if mapping_df is None or mapping_df.empty:
        raise ValueError("A non-empty feeder mapping CSV is required.")

    missing = sorted(set(MAPPING_REQUIRED) - set(mapping_df.columns))
    if missing:
        raise ValueError(f"Feeder mapping missing columns: {missing}")

    data = events.copy()
    mapping = mapping_df.copy()

    if "feeder_name" not in data.columns:
        if "feeder" in data.columns:
            data = data.rename(columns={"feeder": "feeder_name"})
        else:
            raise ValueError("Events need a feeder_name or feeder column.")

    data["_feeder_key"] = data["feeder_name"].map(normalize_mapping_key)
    mapping["_feeder_key"] = mapping["feeder_name"].map(normalize_mapping_key)

    if mapping["_feeder_key"].eq("").any():
        raise ValueError("Mapping contains an empty feeder_name.")

    if mapping["location_id"].isna().any():
        raise ValueError("Mapping contains an empty location_id.")

    if mapping["_feeder_key"].duplicated().any():
        duplicates = mapping.loc[
            mapping["_feeder_key"].duplicated(keep=False), "feeder_name"
        ].tolist()
        raise ValueError(f"Duplicate normalized feeder mapping keys: {duplicates}")

    mapping["location_id"] = mapping["location_id"].astype("string").str.strip()

    # Preserve optional mapping metadata when available.
    keep_columns = ["_feeder_key", "location_id"] + [
        col for col in ["latitude", "longitude", "area"]
        if col in mapping.columns
    ]
    result = data.merge(
        mapping[keep_columns],
        on="_feeder_key",
        how="left",
        validate="many_to_one",
    )

    unmapped = result[result["location_id"].isna()].copy()
    mapped = result[result["location_id"].notna()].copy()

    mapped = mapped.drop(columns=["_feeder_key"], errors="ignore")
    unmapped = unmapped.drop(columns=["_feeder_key"], errors="ignore")

    return mapped.reset_index(drop=True), unmapped.reset_index(drop=True)
