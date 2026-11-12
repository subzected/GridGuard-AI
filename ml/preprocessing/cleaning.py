
import pandas as pd


def clean_dataset(df, numeric_features, max_missing_fraction=0.30):
    df = df.copy()

    df = df.dropna(
        subset=["location_id", "snapshot_time", "label"]
    )

    df["snapshot_time"] = pd.to_datetime(
        df["snapshot_time"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["snapshot_time"])

    df["label"] = pd.to_numeric(
        df["label"], errors="coerce"
    )

    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    for column in numeric_features:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        )

    missing_fraction = (
        df[numeric_features].isna().mean(axis=1)
    )

    df = df[
        missing_fraction <= max_missing_fraction
    ].copy()

    df = df.sort_values(
        ["snapshot_time", "location_id"]
    )

    df = df.drop_duplicates(
        ["location_id", "snapshot_time"],
        keep="last",
    )

    return df.reset_index(drop=True)
