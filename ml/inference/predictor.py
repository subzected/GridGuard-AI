
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
import yaml

from ml.inference.feature_pipeline import (
    prepare_single_weather,
)


ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(ROOT / "ml" / "config.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def predict_outage(
    location_id,
    weather,
    snapshot_time,
    horizon="15m",
):
    config = load_config()

    if horizon not in config["prediction"]["horizons"]:
        raise ValueError("Invalid prediction horizon.")

    model_path = (
        ROOT / config["paths"]["models"]
        / f"gridguard_{horizon}.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    snapshot_time = pd.Timestamp(snapshot_time)

    if snapshot_time.tzinfo is None:
        raise ValueError(
            "snapshot_time must include a timezone."
        )

    snapshot_time = snapshot_time.tz_convert("UTC")

    now = pd.Timestamp(datetime.now(timezone.utc))

    age_minutes = (
        now - snapshot_time
    ).total_seconds() / 60

    max_age = config["inference"]["stale_data_minutes"]

    if age_minutes < -5:
        raise ValueError("Weather snapshot is in the future.")

    if age_minutes > max_age:
        raise ValueError("Weather snapshot is stale.")

    artifact = joblib.load(model_path)

    X = prepare_single_weather(
        weather,
        location_id,
        config,
    )

    model = artifact["model"]

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    threshold = float(artifact["threshold"])

    return {
        "location_id": str(location_id),
        "horizon": horizon,
        "horizon_minutes": config["prediction"]["horizons"][horizon],
        "outage_probability": round(probability, 4),
        "threshold": round(threshold, 4),
        "predicted_outage": probability >= threshold,
        "model_version": artifact["model_version"],
        "snapshot_time": snapshot_time.isoformat(),
        "data_age_minutes": round(age_minutes, 2),
    }
