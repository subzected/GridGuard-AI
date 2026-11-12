"""Structural validation for raw CESC scraped outage records."""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "event_id",
    "source",
    "feeder",
    "start_time",
    "event_type",
]

OPTIONAL_COLUMNS = [
    "source_event_id",
    "affected_area",
    "end_time",
    "reason",
    "verified",
    "label_eligible",
    "source_url",
    "raw_text",
    "duration_minutes",
    "voltage_category",
    "load_outage_kw",
    "load_restoration_kw",
    "energy_loss",
]


def validate_raw_events(df: pd.DataFrame):
    """
    Validate the structure of a scraped CESC dataframe.

    Returns
    -------
    valid_df, rejected_df, report
        Rows with missing start time or feeder are rejected from the
        event-processing path, but retained in rejected_df for audit.
        A missing end_time is allowed because an outage may still be active.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("The raw outage dataframe is empty.")

    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            f"Raw CESC file is missing required columns: {missing_columns}"
        )

    # Ensure optional fields exist so later stages have a predictable schema.
    for column in OPTIONAL_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    data["source"] = data["source"].astype("string").str.strip()
    data["feeder"] = data["feeder"].astype("string").str.strip()
    data["event_type"] = data["event_type"].astype("string").str.strip()

    reasons = [[] for _ in range(len(data))]

    def add_reason(mask, message):
        for position, flag in enumerate(mask.fillna(False).tolist()):
            if flag:
                reasons[position].append(message)

    # Scrapers sometimes turn date headings into rows. They generally have
    # no source event ID, no valid timestamp and a date in the feeder column.
    start_parsed = pd.to_datetime(data["start_time"], errors="coerce", utc=True)
    add_reason(start_parsed.isna(), "missing_or_invalid_start_time")

    feeder_missing = data["feeder"].isna() | data["feeder"].str.strip().eq("")
    add_reason(feeder_missing, "missing_feeder")

    source_missing = data["source"].isna() | data["source"].str.strip().eq("")
    add_reason(source_missing, "missing_source")

    # A non-empty event_id should identify one source row. Duplicate rows
    # are rejected after the first occurrence; source_event_id is also checked.
    event_id = data["event_id"].astype("string").str.strip()
    duplicate_event = event_id.notna() & event_id.ne("") & event_id.duplicated(keep="first")
    add_reason(duplicate_event, "duplicate_event_id")

    source_id = data["source_event_id"].astype("string").str.strip()
    duplicate_source_id = (
        source_id.notna()
        & source_id.ne("")
        & source_id.duplicated(keep="first")
    )
    add_reason(duplicate_source_id, "duplicate_source_event_id")

    # Reject impossible end-before-start timestamps. Parse as UTC for this
    # structural comparison; final timezone interpretation occurs in normalizer.
    end_parsed = pd.to_datetime(data["end_time"], errors="coerce", utc=True)
    impossible_end = (
        start_parsed.notna()
        & end_parsed.notna()
        & (end_parsed < start_parsed)
    )
    add_reason(impossible_end, "end_time_before_start_time")

    data["rejection_reasons"] = [
        ";".join(items) if items else ""
        for items in reasons
    ]

    rejected = data[data["rejection_reasons"].ne("")].copy()
    valid = data[data["rejection_reasons"].eq("")].copy()

    report = {
        "input_rows": int(len(data)),
        "valid_rows": int(len(valid)),
        "rejected_rows": int(len(rejected)),
        "missing_or_invalid_start_time": int(start_parsed.isna().sum()),
        "missing_feeder": int(feeder_missing.sum()),
        "missing_source": int(source_missing.sum()),
        "duplicate_event_id": int(duplicate_event.sum()),
        "duplicate_source_event_id": int(duplicate_source_id.sum()),
        "end_time_before_start_time": int(impossible_end.sum()),
    }

    return valid.reset_index(drop=True), rejected.reset_index(drop=True), report
