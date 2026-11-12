
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import average_precision_score
from sklearn.base import clone

from scipy.stats import randint, uniform


PARAMETER_SPACE = {
    "random_forest": {
        "classifier__n_estimators": randint(200, 601),
        "classifier__max_depth": [6, 8, 10, 14, None],
        "classifier__min_samples_leaf": randint(2, 12),
        "classifier__max_features": ["sqrt", 0.7, 1.0],
    },

    "logistic_regression": {
        "classifier__C": uniform(0.01, 10.0),
        "classifier__class_weight": [
            "balanced", None
        ],
    },

    "hist_gradient_boosting": {
        "classifier__max_iter": randint(100, 401),
        "classifier__learning_rate": uniform(0.02, 0.18),
        "classifier__max_leaf_nodes": randint(7, 32),
        "classifier__l2_regularization": uniform(0.0, 3.0),
    },
}


def tune_model(
    model_name,
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    trials=20,
    random_state=42,
):
    if model_name not in PARAMETER_SPACE:
        raise ValueError(
            f"No parameter space for {model_name}"
        )

    configurations = list(ParameterSampler(
        PARAMETER_SPACE[model_name],
        n_iter=trials,
        random_state=random_state,
    ))

    best_score = -1
    best_model = None
    best_params = None

    for params in configurations:
        candidate = clone(model)
        candidate.set_params(**params)

        candidate.fit(X_train, y_train)

        probabilities = candidate.predict_proba(
            X_validation
        )[:, 1]

        score = average_precision_score(
            y_validation, probabilities
        )

        if score > best_score:
            best_score = score
            best_model = candidate
            best_params = params

    return {
        "model": best_model,
        "best_score": float(best_score),
        "best_params": best_params,
    }
