"""Normalize CESC event fields and timestamps."""

from __future__ import annotations

import re
import pandas as pd


def normalize_feeder_name(value):
    """Normalize spacing and case without guessing feeder identity."""
    if pd.isna(value):
        return pd.NA
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value.upper() if value else pd.NA


def _to_utc(series: pd.Series, naive_timezone: str) -> pd.Series:
    """
    Convert timestamps to UTC.

    Naive timestamps are interpreted in naive_timezone. For the supplied
    CESC CSV, parsed timestamps such as 05:29 correspond to UTC while the
    raw_text contains 10:59 Kolkata time. Therefore the default is UTC.
    Change this only if your input CSV contains unconverted Kolkata times.
    """
    parsed = pd.to_datetime(series, errors="coerce")

    def convert_one(value):
        if pd.isna(value):
            return pd.NaT
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(
                naive_timezone,
                ambiguous="NaT",
                nonexistent="NaT",
            )
        return stamp.tz_convert("UTC")

    return parsed.map(convert_one)


def normalize_events(
    df: pd.DataFrame,
    naive_timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Standardize names, timestamps and numeric fields.

    Does not change verification status and does not infer feeder location IDs.
    """
    if df.empty:
        return df.copy()

    data = df.copy()

    rename_map = {
        "feeder": "feeder_name",
    }
    data = data.rename(columns=rename_map)

    if "feeder_name" not in data.columns:
        raise ValueError("Expected 'feeder' or 'feeder_name' column.")

    for column in ["start_time", "end_time"]:
        if column not in data.columns:
            data[column] = pd.NaT
        data[column] = _to_utc(data[column], naive_timezone)

    data["feeder_name"] = data["feeder_name"].map(normalize_feeder_name)
    data["source"] = data["source"].astype("string").str.strip().str.upper()

    if "event_type" not in data.columns:
        data["event_type"] = "unknown"
    data["source_event_type"] = (
        data["event_type"].astype("string").str.strip().str.lower()
    )

    numeric_columns = [
        "duration_minutes",
        "load_outage_kw",
        "load_restoration_kw",
        "energy_loss",
    ]
    for column in numeric_columns:
        if column not in data.columns:
            data[column] = pd.NA
        data[column] = pd.to_numeric(data[column], errors="coerce")

    calculated_duration = (
        (data["end_time"] - data["start_time"]).dt.total_seconds() / 60.0
    )
    data["duration_minutes_source"] = data["duration_minutes"]
    data["duration_minutes"] = data["duration_minutes"].where(
        data["duration_minutes"].notna(),
        calculated_duration,
    )

    data["event_id"] = data["event_id"].astype("string").str.strip()
    data["source_event_id"] = (
        data["source_event_id"].astype("string").str.strip()
    )
    data["affected_area"] = data["affected_area"].astype("string").str.strip()

    # Keep original source timestamps if present for traceability.
    if "raw_text" not in data.columns:
        data["raw_text"] = pd.NA

    return data.reset_index(drop=True)
