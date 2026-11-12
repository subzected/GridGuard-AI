import pandas as pd


def build_coverage_index(coverage):
    result = {}

    for location, group in coverage.groupby("location_id"):
        result[location] = sorted(
            zip(
                group["coverage_start"],
                group["coverage_end"],
            )
        )

    return result


def is_fully_covered(start, end, intervals):
    """Check whether intervals continuously cover [start, end)."""

    cursor = start

    for interval_start, interval_end in intervals:
        if interval_end <= cursor:
            continue

        if interval_start > cursor:
            return False

        cursor = max(cursor, interval_end)

        if cursor >= end:
            return True

    return False


def generate_labels(
    weather,
    events,
    coverage,
    horizon_minutes,
):
    coverage_index = build_coverage_index(coverage)

    event_index = {
        location: group.sort_values("start_time")
        for location, group in events.groupby("location_id")
    }

    records = []

    for row in weather.itertuples(index=False):
        location = row.location_id
        snapshot = row.snapshot_time

        target_end = (
            snapshot + pd.Timedelta(minutes=horizon_minutes)
        )

        location_events = event_index.get(location)

        if location_events is None:
            continue

        # Exclude snapshots during an active outage.
        active = (
            (location_events["start_time"] <= snapshot)
            & (
                location_events["end_time"].isna()
                | (location_events["end_time"] > snapshot)
            )
        )

        if active.any():
            continue

        intervals = coverage_index.get(location, [])

        if not is_fully_covered(
            snapshot, target_end, intervals
        ):
            continue

        # Outage starts in (snapshot, target_end].
        future = (
            (location_events["start_time"] > snapshot)
            & (location_events["start_time"] <= target_end)
        )

        record = row._asdict()
        record["label"] = int(future.any())
        record["horizon_minutes"] = horizon_minutes

        records.append(record)

    result = pd.DataFrame(records)

    if result.empty:
        raise ValueError(
            f"No valid labels for {horizon_minutes} minutes."
        )

    return result
