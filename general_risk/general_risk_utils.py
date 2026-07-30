"""
General-audience diabetes risk screening.

Unlike the retina-based models (which need specialist equipment -- a fundus
camera), this uses simple health metrics that any individual can typically
provide themselves, similar in spirit to validated screening tools like the
ADA Diabetes Risk Test or FINDRISC. Trained on the Pima Indians Diabetes
Dataset using the same multi-model approach as the rest of this project.

This is a SEPARATE, simpler screening tool -- it estimates general diabetes
risk, not diabetic retinopathy specifically, and is not a substitute for the
retina-based models or a real medical test.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

_DIR = os.path.dirname(__file__)

_MODELS = None
_SCALER = None
_FEATURES = None
_RESULTS = None


def _ensure_loaded():
    global _MODELS, _SCALER, _FEATURES, _RESULTS
    if _MODELS is None:
        _MODELS = joblib.load(os.path.join(_DIR, "trained_models.joblib"))
        _SCALER = joblib.load(os.path.join(_DIR, "scaler.joblib"))
        with open(os.path.join(_DIR, "feature_columns.json")) as f:
            _FEATURES = json.load(f)
        with open(os.path.join(_DIR, "model_results.json")) as f:
            _RESULTS = json.load(f)


def get_feature_list():
    _ensure_loaded()
    return _FEATURES


def get_model_results():
    _ensure_loaded()
    return _RESULTS


def predict_risk(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    """Run the ensemble + individual models on the given health metrics."""
    _ensure_loaded()

    input_df = pd.DataFrame(
        [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
        columns=_FEATURES,
    )
    scaled = _SCALER.transform(input_df)

    per_model = {}
    for name, model in _MODELS.items():
        prob = float(model.predict_proba(scaled)[0][1])
        per_model[name] = {
            "probability": prob,
            "prediction": "Elevated risk" if prob > 0.5 else "Lower risk",
        }

    ensemble_prob = per_model["Ensemble (Voting)"]["probability"]
    return {
        "per_model": per_model,
        "ensemble_probability": ensemble_prob,
        "ensemble_verdict": "Elevated risk" if ensemble_prob > 0.5 else "Lower risk",
    }
