
import pandas as pd


def chronological_split(
    df,
    train_fraction=0.60,
    validation_fraction=0.15,
    calibration_fraction=0.10,
    test_fraction=0.15,
    embargo_minutes=60,
):
    total = (
        train_fraction
        + validation_fraction
        + calibration_fraction
        + test_fraction
    )

    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.")

    df = df.sort_values("snapshot_time").reset_index(drop=True)

    timestamps = sorted(df["snapshot_time"].unique())

    if len(timestamps) < 20:
        raise ValueError("Insufficient unique timestamps.")

    n = len(timestamps)

    t1 = timestamps[int(n * train_fraction)]
    t2 = timestamps[
        int(n * (train_fraction + validation_fraction))
    ]
    t3 = timestamps[
        int(n * (
            train_fraction
            + validation_fraction
            + calibration_fraction
        ))
    ]

    embargo = pd.Timedelta(minutes=embargo_minutes)

    splits = {
        "train": df[df["snapshot_time"] < t1],

        "validation": df[
            (df["snapshot_time"] >= t1 + embargo)
            & (df["snapshot_time"] < t2)
        ],

        "calibration": df[
            (df["snapshot_time"] >= t2 + embargo)
            & (df["snapshot_time"] < t3)
        ],

        "test": df[
            df["snapshot_time"] >= t3 + embargo
        ],
    }

    for name, split in splits.items():
        if split.empty:
            raise ValueError(f"{name} split is empty.")

        if split["label"].nunique() < 2:
            raise ValueError(
                f"{name} split does not contain both classes."
            )

    return splits


def validate_split_sizes(splits, minimum_samples=100):
    for name, split in splits.items():
        if len(split) < minimum_samples:
            raise ValueError(
                f"{name} has only {len(split)} samples."
            )

        print(
            f"{name}: {len(split)} samples | "
            f"positive: {split['label'].sum()} | "
            f"rate: {split['label'].mean():.3f}"
        )
