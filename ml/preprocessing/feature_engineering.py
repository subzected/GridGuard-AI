import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate deterministic, row-level weather features.

    This function intentionally does not calculate rolling or lagged
    features. Those require ordered historical observations and are
    implemented separately in the historical feature pipeline.
    """
    df = df.copy()

    required_inputs = [
        "temp_c",
        "humidity_pct",
        "precipitation_mm",
        "wind_speed_kmh",
        "wind_gusts_kmh",
        "wind_direction_deg",
    ]

    missing = [column for column in required_inputs if column not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot engineer weather features; missing columns: {missing}"
        )

    numeric_inputs = required_inputs + [
        "rain_mm",
        "pressure_hpa",
        "precipitation_probability_pct",
        "thunderstorm_probability_pct",
    ]

    for column in numeric_inputs:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace infinite values with NaN; do not invent replacement values.
    present_numeric = [column for column in numeric_inputs if column in df.columns]
    df[present_numeric] = df[present_numeric].replace(
        [np.inf, -np.inf], np.nan
    )

    # Wind gust intensity.
    df["wind_gust_delta"] = (
        df["wind_gusts_kmh"] - df["wind_speed_kmh"]
    ).clip(lower=0)

    # Circular encoding of wind direction.
    direction = np.deg2rad(df["wind_direction_deg"] % 360)
    df["wind_direction_sin"] = np.sin(direction)
    df["wind_direction_cos"] = np.cos(direction)

    # Temperature-humidity interaction (retains the existing project formula).
    df["temperature_humidity_index"] = (
        0.8 * df["temp_c"]
        + (df["humidity_pct"] / 100.0) * (df["temp_c"] - 14.4)
        + 46.4
    )

    # Precipitation and wind interaction.
    df["precipitation_wind_interaction"] = (
        df["precipitation_mm"] * df["wind_speed_kmh"]
    )

    derived = [
        "wind_gust_delta",
        "wind_direction_sin",
        "wind_direction_cos",
        "temperature_humidity_index",
        "precipitation_wind_interaction",
    ]
    df[derived] = df[derived].replace([np.inf, -np.inf], np.nan)

    return df


def get_feature_columns(config):
    """Return model feature columns in the configured order."""
    features = config["features"]
    return (
        features["numeric"]
        + features["derived"]
        + features["categorical"]
    )
