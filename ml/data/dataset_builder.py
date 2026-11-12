from pathlib import Path

import pandas as pd
import yaml

from ml.data.quality_checks import (
    validate_weather,
    validate_events,
    validate_coverage,
)

from ml.data.label_generator import generate_labels


ROOT = Path(__file__).resolve().parents[2]


def load_config():
    path = ROOT / "ml" / "config.yaml"

    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_dataset():
    config = load_config()

    raw_dir = ROOT / config["paths"]["raw_data"]
    processed_dir = ROOT / config["paths"]["processed_data"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    weather = pd.read_csv(
        raw_dir / config["files"]["weather"]
    )

    events = pd.read_csv(
        raw_dir / config["files"]["events"]
    )

    coverage = pd.read_csv(
        raw_dir / config["files"]["coverage"]
    )

    weather = validate_weather(weather)
    events = validate_events(events)
    coverage = validate_coverage(coverage)

    results = []

    horizons = config["prediction"]["horizons"]

    for name, minutes in horizons.items():

        labelled = generate_labels(
            weather,
            events,
            coverage,
            minutes,
        )

        labelled["horizon"] = name

        output = processed_dir / f"dataset_{name}.csv"

        labelled.to_csv(output, index=False)

        print(
            f"{name}: {len(labelled)} rows | "
            f"{labelled['label'].sum()} positive"
        )

        results.append(labelled)

    combined = pd.concat(results, ignore_index=True)

    combined.to_csv(
        processed_dir / config["files"]["dataset"],
        index=False,
    )

    print("Dataset generation completed.")

    return combined


if __name__ == "__main__":
    build_dataset()
