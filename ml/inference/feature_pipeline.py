from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from ml.preprocessing.feature_engineering import (
    engineer_features,
    get_feature_columns,
)


# Database/API names mapped to the canonical names used by the ML model.
COLUMN_ALIASES = {
    "temperature_c": "temp_c",
    "relative_humidity_pct": "humidity_pct",
    "valid_time": "forecast_valid_time",
}


def normalize_weather_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize weather column names without changing timestamp semantics.

    fetched_at = when the weather record was retrieved.
    valid_time / forecast_valid_time = when the forecast applies.
    snapshot_time = when the prediction snapshot was formed.

    This function does not infer or manufacture snapshot_time.
    """
    df = df.copy()

    for source, target in COLUMN_ALIASES.items():
        if source not in df.columns:
            continue

        if target in df.columns:
            # Prefer the canonical column, but fill its missing values
            # from the alias rather than silently overwriting it.
            df[target] = df[target].combine_first(df[source])
            df = df.drop(columns=[source])
        else:
            df = df.rename(columns={source: target})

    if "location_id" in df.columns:
        df["location_id"] = df["location_id"].astype("string").str.strip()

    # Parse timestamps while preserving timezone information.
    for column in ("snapshot_time", "forecast_valid_time", "fetched_at"):
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce", utc=True)

    return df


def _validate_configured_features(df: pd.DataFrame, config: dict) -> list[str]:
    required = get_feature_columns(config)
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            "Missing configured model features after schema normalization: "
            f"{missing}. Check the weather input schema and ml/config.yaml."
        )

    return required


def prepare_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Prepare a batch of weather rows for model input.

    The same function must be used for training and inference.
    Missing numeric values are preserved for the model's fitted imputer,
    if one exists. This function does not fit imputers or encoders.
    """
    if df.empty:
        raise ValueError("Cannot prepare model features from an empty DataFrame.")

    df = normalize_weather_schema(df)
    df = engineer_features(df)

    required = _validate_configured_features(df, config)
    X = df[required].copy()

    categorical = config["features"]["categorical"]
    numeric = config["features"]["numeric"] + config["features"]["derived"]

    for column in numeric:
        X[column] = pd.to_numeric(X[column], errors="coerce")
        X[column] = X[column].replace([np.inf, -np.inf], np.nan)

    for column in categorical:
        # Keep missing categories as missing; do not turn them into the
        # literal string "nan".
        X[column] = X[column].astype("string")

    return X


def prepare_single_weather(
    weather: Mapping,
    location_id: str,
    config: dict,
) -> pd.DataFrame:
    """
    Prepare one live weather record.

    This only creates row-level features. Historical rolling features
    must be supplied by a separate, time-aware context builder.
    """
    if not isinstance(weather, Mapping):
        raise TypeError("weather must be a mapping/dictionary.")

    row = dict(weather)
    row["location_id"] = str(location_id)

    df = pd.DataFrame([row])
    return prepare_features(df, config)
