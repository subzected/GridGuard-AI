"""Apply explicit human/independent-source verification decisions."""

from __future__ import annotations

import pandas as pd


REVIEW_REQUIRED = [
    "event_id",
    "verification_status",
    "verification_method",
]


def apply_verification(
    events: pd.DataFrame,
    review_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Apply a separately maintained review file.

    Accepted verification_status values: confirmed, pending, rejected.
    A confirmed decision must include a method and either an evidence URL
    or a written verification note. Only confirmed + unplanned events can
    become label-eligible. Feeder mapping is checked separately.

    The scraper's `verified` and `label_eligible` columns are ignored.
    """
    data = events.copy()

    if data.empty:
        data["verification_status"] = pd.Series(dtype="string")
        data["verification_method"] = pd.Series(dtype="string")
        data["verification_evidence_url"] = pd.Series(dtype="string")
        data["verification_notes"] = pd.Series(dtype="string")
        data["verified"] = pd.Series(dtype=bool)
        data["label_eligible"] = pd.Series(dtype=bool)
        return data

    data["verification_status"] = "pending"
    data["verification_method"] = "not_reviewed"
    data["verification_evidence_url"] = pd.NA
    data["verification_notes"] = "No explicit review decision supplied."

    if review_df is not None and not review_df.empty:
        missing = sorted(set(REVIEW_REQUIRED) - set(review_df.columns))
        if missing:
            raise ValueError(f"Review file missing columns: {missing}")

        review = review_df.copy()
        review["event_id"] = review["event_id"].astype("string").str.strip()
        review["verification_status"] = (
            review["verification_status"].astype("string").str.strip().str.lower()
        )

        allowed = {"confirmed", "pending", "rejected"}
        invalid = ~review["verification_status"].isin(allowed)
        if invalid.any():
            bad = sorted(review.loc[invalid, "verification_status"].dropna().unique())
            raise ValueError(f"Invalid verification_status values: {bad}")

        if review["event_id"].duplicated().any():
            raise ValueError("Review file contains duplicate event_id values.")

        for optional in ["evidence_url", "verification_notes"]:
            if optional not in review.columns:
                review[optional] = pd.NA

        review = review.rename(
            columns={"evidence_url": "verification_evidence_url"}
        )

        data = data.drop(
            columns=[
                "verification_status",
                "verification_method",
                "verification_evidence_url",
                "verification_notes",
            ],
            errors="ignore",
        ).merge(
            review[
                [
                    "event_id",
                    "verification_status",
                    "verification_method",
                    "verification_evidence_url",
                    "verification_notes",
                ]
            ],
            on="event_id",
            how="left",
            validate="one_to_one",
        )

        data["verification_status"] = data["verification_status"].fillna("pending")
        data["verification_method"] = data["verification_method"].fillna("not_reviewed")
        data["verification_notes"] = data["verification_notes"].fillna(
            "No explicit review decision supplied."
        )

    method_present = (
        data["verification_method"].notna()
        & data["verification_method"].astype(str).str.strip().ne("")
        & data["verification_method"].astype(str).str.lower().ne("not_reviewed")
    )
    url_present = (
        data["verification_evidence_url"].notna()
        & data["verification_evidence_url"].astype(str).str.strip().ne("")
    )
    notes_present = (
        data["verification_notes"].notna()
        & data["verification_notes"].astype(str).str.strip().ne("")
        & data["verification_notes"].astype(str).ne(
            "No explicit review decision supplied."
        )
    )

    supported_confirmation = (
        data["verification_status"].eq("confirmed")
        & method_present
        & (url_present | notes_present)
    )

    # A bare "confirmed" checkbox without an evidence trail is not accepted.
    unsupported_confirmation = data["verification_status"].eq("confirmed") & (
        ~supported_confirmation
    )
    data.loc[unsupported_confirmation, "verification_status"] = "pending"
    data.loc[unsupported_confirmation, "verification_notes"] = (
        "Confirmation lacks a verification method and evidence URL or notes."
    )

    data["verified"] = supported_confirmation
    data["label_eligible"] = (
        data["verified"]
        & data["event_type"].eq("unplanned")
        & data["start_time"].notna()
    )

    return data.reset_index(drop=True)
