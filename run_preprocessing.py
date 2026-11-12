"""Run the complete CESC preprocessing workflow.

Example:
python run_preprocessing.py ^
  --raw data/raw/cesc_outage_raw.csv ^
  --mapping data/raw/feeder_mapping.csv ^
  --review data/raw/cesc_event_review.csv ^
  --output data/processed/cesc

The --review argument is optional. Without it, all events remain pending and
no event is label-eligible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.data.preprocessing.raw_data_validator import validate_raw_events
from ml.data.preprocessing.event_normalizer import normalize_events
from ml.data.preprocessing.event_classifier import classify_events
from ml.data.preprocessing.event_verifier import apply_verification
from ml.data.preprocessing.feeder_mapper import map_feeders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw scraped CESC CSV")
    parser.add_argument("--mapping", required=True, help="Feeder mapping CSV")
    parser.add_argument("--review", default=None, help="Optional human review CSV")
    parser.add_argument(
        "--output",
        default="data/processed/cesc",
        help="Output directory",
    )
    parser.add_argument(
        "--naive-timezone",
        default="UTC",
        help="Timezone for naive timestamps; default UTC for supplied parsed CESC data",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw)
    mapping_path = Path(args.mapping)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(raw_path)
    mapping = pd.read_csv(mapping_path)

    valid, rejected, validation_report = validate_raw_events(raw)
    rejected.to_csv(output_dir / "rejected_records.csv", index=False)

    normalized = normalize_events(valid, naive_timezone=args.naive_timezone)
    normalized.to_csv(output_dir / "normalized_events.csv", index=False)

    classified = classify_events(normalized)
    classified.to_csv(output_dir / "classified_events.csv", index=False)

    review = pd.read_csv(args.review) if args.review else None
    verified = apply_verification(classified, review)
    verified.to_csv(output_dir / "verified_events.csv", index=False)

    mapped, unmapped = map_feeders(verified, mapping)
    unmapped.to_csv(output_dir / "unmapped_feeders.csv", index=False)

    # Standardized file consumed by the current quality_checks.py and
    # label_generator.py. It contains only confirmed unplanned events whose
    # feeder was mapped to a GridGuard location ID.
    eligible = mapped[mapped["label_eligible"]].copy()
    standard_columns = [
        "location_id",
        "start_time",
        "end_time",
        "source",
        "verified",
        "event_id",
        "source_event_id",
        "event_type",
        "feeder_name",
        "affected_area",
        "source_url",
        "label_eligible",
        "verification_method",
        "verification_evidence_url",
    ]
    for col in standard_columns:
        if col not in eligible.columns:
            eligible[col] = pd.NA

    eligible[standard_columns].to_csv(
        output_dir / "outage_events.csv",
        index=False,
    )

    quarantine = mapped[~mapped["label_eligible"]].copy()
    quarantine.to_csv(output_dir / "quarantined_events.csv", index=False)

    report = {
        "raw_file": str(raw_path),
        "mapping_file": str(mapping_path),
        "review_file": str(args.review) if args.review else None,
        "validation": validation_report,
        "normalized_rows": int(len(normalized)),
        "classified_rows": int(len(classified)),
        "mapped_rows": int(len(mapped)),
        "unmapped_rows": int(len(unmapped)),
        "confirmed_rows": int(mapped["verified"].sum()) if len(mapped) else 0,
        "label_eligible_rows": int(len(eligible)),
        "quarantined_rows": int(len(quarantine)),
        "naive_timestamp_timezone": args.naive_timezone,
        "note": (
            "No event is verified automatically. label_eligible requires an "
            "explicit confirmed review, unplanned classification and feeder mapping."
        ),
    }

    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print(json.dumps(report, indent=2, default=str))
    print(f"\nPreprocessing outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
