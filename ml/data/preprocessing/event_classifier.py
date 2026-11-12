"""Classify event type using source declarations and explicit text evidence.

Classification is not verification. An event classified as unplanned remains
pending until a human or trusted independent source confirms it.
"""

from __future__ import annotations

import re
import pandas as pd


PLANNED_PATTERNS = [
    r"\bscheduled\b",
    r"\bplanned\b",
    r"\bmaintenance\b",
    r"\bshutdown\b",
    r"\bwork\s+will\s+be\s+carried\s+out\b",
    r"\bpower\s+interruption\s+notice\b",
]

UNPLANNED_PATTERNS = [
    r"\bunplanned\b",
    r"\bunexpected\b",
    r"\bforced\s+outage\b",
    r"\bfault\b",
    r"\bbreakdown\b",
    r"\btripping\b",
    r"\btechnical\s+fault\b",
]


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def classify_one(source_event_type, reason, raw_text):
    """Return (classification, basis, confidence); confidence is heuristic."""
    source_value = (
        "" if pd.isna(source_event_type) else str(source_event_type).strip().lower()
    )
    text = " ".join(
        str(value) for value in [reason, raw_text]
        if value is not None and not pd.isna(value)
    ).lower()

    planned_text = _contains_any(text, PLANNED_PATTERNS)
    unplanned_text = _contains_any(text, UNPLANNED_PATTERNS)

    if source_value in {"planned", "scheduled", "maintenance"}:
        return "planned", "source_declared", 0.80

    if source_value in {"unplanned", "unscheduled", "forced"}:
        return "unplanned", "source_declared", 0.80

    if planned_text and not unplanned_text:
        return "planned", "text_keyword", 0.60

    if unplanned_text and not planned_text:
        return "unplanned", "text_keyword", 0.60

    return "unknown", "insufficient_or_conflicting_evidence", 0.0


def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add event classification fields without changing verification fields."""
    data = df.copy()
    if data.empty:
        return data

    for column in ["source_event_type", "reason", "raw_text"]:
        if column not in data.columns:
            data[column] = pd.NA

    classifications = data.apply(
        lambda row: classify_one(
            row["source_event_type"],
            row["reason"],
            row["raw_text"],
        ),
        axis=1,
    )

    data["event_type"] = [item[0] for item in classifications]
    data["classification_basis"] = [item[1] for item in classifications]
    data["classification_confidence"] = [item[2] for item in classifications]

    # Do not infer verified=True from event_type.
    data["verified"] = False
    data["label_eligible"] = False
    data["verification_status"] = "pending"

    return data
