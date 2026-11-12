
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def get_model_candidates(
    numeric_features,
    categorical_features,
    random_state=42,
):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }

    candidates = {}

    for name, estimator in models.items():
        candidates[name] = Pipeline([
            (
                "preprocessing",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            ("classifier", estimator),
        ])

    return candidates


def compare_models(candidates, X_train, y_train, X_val, y_val):
    from sklearn.metrics import average_precision_score

    results = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_val)[:, 1]

        score = average_precision_score(
            y_val, probabilities
        )

        results[name] = {
            "model": model,
            "validation_pr_auc": float(score),
        }

        print(f"{name}: PR-AUC={score:.4f}")

    best_name = max(
        results,
        key=lambda name: results[name]["validation_pr_auc"],
    )

    return best_name, results
