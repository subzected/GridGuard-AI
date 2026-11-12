
import pandas as pd

from sklearn.inspection import permutation_importance


def calculate_feature_importance(
    model,
    X,
    y,
    feature_names,
    repeats=10,
    random_state=42,
):
    result = permutation_importance(
        model,
        X,
        y,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    })

    return importance.sort_values(
        "importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def save_feature_importance(importance, output_path):
    importance.to_csv(output_path, index=False)
