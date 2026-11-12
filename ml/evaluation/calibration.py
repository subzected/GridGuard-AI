
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


def calibrate_model(model, X_calibration, y_calibration, method="sigmoid"):
    if method not in {"sigmoid", "isotonic"}:
        raise ValueError("Unsupported calibration method.")

    if len(set(y_calibration)) < 2:
        raise ValueError(
            "Calibration data must contain both classes."
        )

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(model),
        method=method,
    )

    calibrated.fit(X_calibration, y_calibration)

    return calibrated
