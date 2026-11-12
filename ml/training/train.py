
from pathlib import Path
from datetime import datetime, timezone

import json
import joblib
import pandas as pd
import yaml

from ml.preprocessing.cleaning import clean_dataset
from ml.inference.feature_pipeline import prepare_features

from ml.training.model_comparison import (
    get_model_candidates,
    compare_models,
)
from ml.training.hyperparameter_tuning import tune_model

from ml.evaluation.validation import (
    chronological_split,
    validate_split_sizes,
)
from ml.evaluation.calibration import calibrate_model
from ml.evaluation.metrics import (
    evaluate_predictions,
    choose_threshold,
)
from ml.evaluation.explainability import (
    calculate_feature_importance,
)


ROOT = Path(__file__).resolve().parents[2]


def load_config():
    with open(
        ROOT / "ml" / "config.yaml",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


def train_one_horizon(df, horizon, config):

    training_config = config["training"]
    feature_config = config["features"]

    # --------------------------------------------------
    # 1. Define feature columns
    # --------------------------------------------------

    numeric_features = (
        feature_config["numeric"]
        + feature_config["derived"]
    )

    categorical_features = feature_config["categorical"]

    feature_names = (
        numeric_features + categorical_features
    )

    # --------------------------------------------------
    # 2. Clean dataset
    # --------------------------------------------------

    labelled = clean_dataset(
        df,
        feature_config["numeric"],
        config["data_quality"]["max_missing_feature_fraction"],
    )

    # --------------------------------------------------
    # 3. Preserve labels and timestamps
    # --------------------------------------------------

    labels = labelled["label"].copy()

    timestamps = labelled["snapshot_time"].copy()

    # --------------------------------------------------
    # 4. Prepare features
    # --------------------------------------------------

    features = prepare_features(
        labelled,
        config,
    )

    # Restore target and timestamp columns
    df = features.copy()

    df["label"] = labels.to_numpy()
    df["snapshot_time"] = timestamps.to_numpy()

    # --------------------------------------------------
    # 5. Chronological train/validation/calibration/test split
    # --------------------------------------------------

    splits = chronological_split(
        df,
        train_fraction=training_config["train_fraction"],
        validation_fraction=training_config["validation_fraction"],
        calibration_fraction=training_config["calibration_fraction"],
        test_fraction=training_config["test_fraction"],
        embargo_minutes=config["prediction"]["embargo_minutes"],
    )

    validate_split_sizes(
        splits,
        minimum_samples=training_config["min_samples"],
    )

    # --------------------------------------------------
    # 6. Validate positive samples in each split
    # --------------------------------------------------

    for name, split in splits.items():

        positive_count = int(split["label"].sum())

        if positive_count < training_config["min_positive_samples"]:

            raise ValueError(
                f"{horizon} {name}: only {positive_count} "
                "positive examples. More verified events needed."
            )

    # --------------------------------------------------
    # 7. Prepare X and y
    # --------------------------------------------------

    X_train = splits["train"][feature_names]
    y_train = splits["train"]["label"]

    X_val = splits["validation"][feature_names]
    y_val = splits["validation"]["label"]

    X_cal = splits["calibration"][feature_names]
    y_cal = splits["calibration"]["label"]

    X_test = splits["test"][feature_names]
    y_test = splits["test"]["label"]

    # --------------------------------------------------
    # 8. Model comparison
    # --------------------------------------------------

    candidates = get_model_candidates(
        numeric_features,
        categorical_features,
        random_state=config["project"]["random_state"],
    )

    best_name, comparison = compare_models(
        candidates,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    selected = comparison[best_name]["model"]

    # --------------------------------------------------
    # 9. Hyperparameter tuning
    # --------------------------------------------------

    if training_config["hyperparameter_tuning"]["enabled"]:

        tuning = tune_model(
            best_name,
            selected,
            X_train,
            y_train,
            X_val,
            y_val,
            trials=training_config["hyperparameter_tuning"]["trials"],
            random_state=config["project"]["random_state"],
        )

        selected = tuning["model"]

        best_params = tuning["best_params"]

    else:

        best_params = {}

    # --------------------------------------------------
    # 10. Refit on development data
    # --------------------------------------------------

    development = pd.concat([
        splits["train"],
        splits["validation"],
    ]).sort_values("snapshot_time")

    X_development = development[feature_names]
    y_development = development["label"]

    selected.fit(
        X_development,
        y_development,
    )

    # --------------------------------------------------
    # 11. Probability calibration
    # --------------------------------------------------

    calibrated = calibrate_model(
        selected,
        X_cal,
        y_cal,
        method=config["calibration"]["method"],
    )

    calibration_probabilities = calibrated.predict_proba(
        X_cal
    )[:, 1]

    # --------------------------------------------------
    # 12. Threshold selection
    # --------------------------------------------------

    threshold_config = config["evaluation"]

    threshold = choose_threshold(
        y_cal,
        calibration_probabilities,
        minimum=threshold_config["threshold_min"],
        maximum=threshold_config["threshold_max"],
        step=threshold_config["threshold_step"],
    )

    # --------------------------------------------------
    # 13. Final test evaluation
    # --------------------------------------------------

    test_probabilities = calibrated.predict_proba(
        X_test
    )[:, 1]

    metrics = evaluate_predictions(
        y_test,
        test_probabilities,
        threshold,
    )

    # --------------------------------------------------
    # 14. Feature importance
    # --------------------------------------------------

    importance = calculate_feature_importance(
        calibrated,
        X_test,
        y_test,
        feature_names,
    )

    # --------------------------------------------------
    # 15. Create output directories
    # --------------------------------------------------

    model_dir = ROOT / config["paths"]["models"]

    report_dir = ROOT / config["paths"]["reports"]

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # 16. Save model artifact
    # --------------------------------------------------

    model_version = (
        f"{horizon}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )

    artifact = {
        "model": calibrated,
        "features": feature_names,
        "horizon": horizon,
        "threshold": threshold,
        "selected_model": best_name,
        "best_params": best_params,
        "model_version": model_version,
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    joblib.dump(
        artifact,
        model_dir / f"gridguard_{horizon}.joblib",
    )

    # --------------------------------------------------
    # 17. Save feature importance
    # --------------------------------------------------

    importance.to_csv(
        report_dir / f"{horizon}_feature_importance.csv",
        index=False,
    )

    # --------------------------------------------------
    # 18. Generate training report
    # --------------------------------------------------

    report = {
        "horizon": horizon,
        "model_version": model_version,
        "selected_model": best_name,
        "best_params": best_params,

        "comparison_scores": {
            name: value["validation_pr_auc"]
            for name, value in comparison.items()
        },

        "threshold": threshold,

        "test_metrics": metrics,

        "split_sizes": {
            name: len(split)
            for name, split in splits.items()
        },
    }

    with open(
        report_dir / f"{horizon}_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            default=str,
        )

    return report


# ======================================================
# MAIN TRAINING PIPELINE
# ======================================================

def main():

    config = load_config()

    processed_dir = ROOT / config["paths"]["processed_data"]

    all_reports = []

    # Train separate models for each prediction horizon
    for horizon in config["prediction"]["horizons"]:

        dataset_path = (
            processed_dir / f"dataset_{horizon}.csv"
        )

        if not dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: {dataset_path}. "
                "Run ml/data/dataset_builder.py first."
            )

        labelled = pd.read_csv(dataset_path)

        # Parse timestamps
        labelled["snapshot_time"] = pd.to_datetime(
            labelled["snapshot_time"],
            utc=True,
            errors="coerce",
        )

        report = train_one_horizon(
            labelled,
            horizon,
            config,
        )

        all_reports.append(report)

    # --------------------------------------------------
    # Save combined training summary
    # --------------------------------------------------

    report_dir = ROOT / config["paths"]["reports"]

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        report_dir / "training_summary.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            all_reports,
            file,
            indent=4,
            default=str,
        )

    print("\nAll horizon models trained successfully.")


if __name__ == "__main__":
    main()
