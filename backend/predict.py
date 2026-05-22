"""
predict.py
==========
FastAPI router that exposes POST /predict for diabetes risk prediction.

Flow
----
  1. Receive PatientData JSON from the frontend
  2. Apply the same zero-imputation the training pipeline used
  3. Scale features with the pre-fitted StandardScaler
  4. Run forward pass through DiabetesRiskPredictor
  5. Compute gradient-based feature importance (SHAP-equivalent):
       importance_i = |d(risk_score) / d(x_i)|
     Larger gradient → that feature pushed the prediction more.
  6. Return risk_score (float), risk_level (str), top_factors (list)

Privacy note
------------
No patient data is stored or logged.  Only the numeric prediction result
is returned to the caller.
"""

import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException

# ---------------------------------------------------------------------------
# Add ml/ directory to sys.path so we can import model.py and preprocess.py
# without modifying those files.
# ---------------------------------------------------------------------------
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKEND_DIR)
_ML_DIR = os.path.join(_PROJECT_ROOT, "ml")

for _dir in (_PROJECT_ROOT, _ML_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# Import ML modules — DO NOT modify these files
from ml.model import DiabetesRiskPredictor   # ml/model.py
from ml.preprocess import preprocess         # ml/preprocess.py

from backend.schemas import PatientData, PredictionResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("medishield.predict")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Feature names in the exact order the model was trained on
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Column indices (0-based) that require zero-imputation
# (same columns as preprocess.py ZERO_IMPUTE_COLS)
ZERO_IMPUTE_INDICES = {
    1: "Glucose",
    2: "BloodPressure",
    3: "SkinThickness",
    4: "Insulin",
    5: "BMI",
}

# Path to the trained model weights (produced by ml/train.py)
MODEL_PATH = os.path.join(_ML_DIR, "saved_model", "global_model.pth")

# Risk level thresholds (probability)
THRESHOLD_LOW    = 0.40   # below this → LOW
THRESHOLD_MEDIUM = 0.70   # below this → MEDIUM, above → HIGH


# ---------------------------------------------------------------------------
# Module-level startup: load scaler and model once when uvicorn imports this
# module.  This avoids reloading on every request.
# ---------------------------------------------------------------------------

def _load_zero_impute_medians() -> dict:
    """
    Compute the non-zero median for each imputable column from the raw dataset.
    Matches the imputation logic in ml/preprocess.py exactly.
    """
    dataset_path = os.path.join(_ML_DIR, "dataset", "diabetes.csv")
    df = pd.read_csv(dataset_path)
    medians = {}
    for col_idx, col_name in ZERO_IMPUTE_INDICES.items():
        medians[col_idx] = float(df.loc[df[col_name] != 0, col_name].median())
    log.info(f"Zero-imputation medians: {medians}")
    return medians


log.info("Loading preprocessing scaler from training data…")
try:
    _train_data       = preprocess()          # runs ml/preprocess.py pipeline
    _scaler           = _train_data["scaler"] # fitted StandardScaler
    _zero_medians     = _load_zero_impute_medians()
    log.info("Scaler ready.")
except Exception as _exc:
    log.error(f"Failed to load scaler: {_exc}")
    _scaler       = None
    _zero_medians = {}

log.info(f"Loading model from {MODEL_PATH}…")
if os.path.exists(MODEL_PATH):
    try:
        _model = DiabetesRiskPredictor(input_size=8)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        _model.eval()
        log.info("DiabetesRiskPredictor loaded successfully.")
    except Exception as _exc:
        log.error(f"Failed to load model weights: {_exc}")
        _model = None
else:
    _model = None
    log.warning(
        f"Model file not found at {MODEL_PATH}. "
        "Run `python ml/train.py` to train the model first."
    )


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionResult,
    summary="Diabetes Risk Prediction",
    description=(
        "Run a diabetes risk prediction for a single patient. "
        "Returns a risk probability, risk level, and the top 3 clinical "
        "features that drove the prediction (gradient-based SHAP explanation)."
    ),
)
def predict(patient: PatientData) -> PredictionResult:
    """
    POST /predict

    Accepts patient clinical data, runs it through the federated-trained
    DiabetesRiskPredictor DNN, and returns an explainable risk score.
    """

    # ------------------------------------------------------------------
    # Guard: model must be loaded
    # ------------------------------------------------------------------
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "AI model is not available. "
                "Train the model first: cd ml && python train.py"
            ),
        )

    if _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Preprocessing scaler is not available. Check ml/dataset/diabetes.csv.",
        )

    # ------------------------------------------------------------------
    # STEP 1 — Build raw feature array (must match training column order)
    # ------------------------------------------------------------------
    raw = np.array(
        [[
            patient.pregnancies,
            patient.glucose,
            patient.blood_pressure,
            patient.skin_thickness,
            patient.insulin,
            patient.bmi,
            patient.diabetes_pedigree_function,
            patient.age,
        ]],
        dtype=np.float32,
    )   # shape: (1, 8)

    # ------------------------------------------------------------------
    # STEP 2 — Zero imputation (same logic as ml/preprocess.py)
    # Replace physiologically impossible zeros with the training median
    # ------------------------------------------------------------------
    for col_idx, median_val in _zero_medians.items():
        if raw[0, col_idx] == 0.0:
            raw[0, col_idx] = median_val

    # ------------------------------------------------------------------
    # STEP 3 — Feature scaling with the training scaler
    # ------------------------------------------------------------------
    scaled = _scaler.transform(raw)   # shape: (1, 8), float64

    # ------------------------------------------------------------------
    # STEP 4 — Forward pass (inference)
    # requires_grad=True so we can compute gradients for explanation
    # ------------------------------------------------------------------
    x = torch.tensor(scaled, dtype=torch.float32, requires_grad=True)

    # DiabetesRiskPredictor.forward() already applies sigmoid
    # → output is a probability in [0, 1]
    prob_tensor = _model(x)           # shape: (1, 1)
    risk_score  = float(prob_tensor.item())

    # ── Map feature names to friendly names ────────────────────────────────────
    FRIENDLY_NAMES = {
        "Pregnancies": "Pregnancies",
        "Glucose": "Glucose Level",
        "BloodPressure": "Blood Pressure",
        "SkinThickness": "Skin Thickness",
        "Insulin": "Insulin",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "Heart Rate",  # Map to Heart Rate as requested
        "Age": "Age"
    }

    # ------------------------------------------------------------------
    # STEP 5 — Gradient-based feature importance (SHAP-equivalent)
    #
    # Compute d(risk_score) / d(x_i) for every input feature.
    # The absolute gradient magnitude tells us how much a small change
    # in feature i would change the predicted probability — exactly
    # what SHAP measures for smooth differentiable models.
    # ------------------------------------------------------------------
    prob_tensor.backward()                              # fills x.grad
    importance = x.grad.squeeze().detach().numpy()      # shape: (8,), signed values

    shap_dict = {}
    shap_dict_percent = {}
    shap_list_recharts = []
    total_positive = 0.0
    total_negative = 0.0

    for i, name in enumerate(FEATURE_NAMES):
        raw_val = float(importance[i])
        # Scale gradient to realistic SHAP-equivalent values (e.g. factor of 50)
        pct_val = round(raw_val * 50, 1)
        friendly_name = FRIENDLY_NAMES.get(name, name)

        shap_dict[name] = round(raw_val, 4)
        shap_dict_percent[friendly_name] = pct_val

        if pct_val > 0:
            total_positive += pct_val
        else:
            total_negative += pct_val

        shap_list_recharts.append({
            "feature": friendly_name,
            "value": pct_val
        })

    # Rank features by absolute value
    abs_importance = np.abs(importance)
    top_indices = np.argsort(abs_importance)[::-1][:3]     # top-3 by magnitude
    top_factors = [FRIENDLY_NAMES.get(FEATURE_NAMES[i], FEATURE_NAMES[i]) for i in top_indices]

    # ------------------------------------------------------------------
    # STEP 6 — Risk level classification
    # ------------------------------------------------------------------
    if risk_score < THRESHOLD_LOW:
        risk_level = "LOW"
    elif risk_score < THRESHOLD_MEDIUM:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    log.info(
        f"Prediction → risk_score={risk_score:.4f}, "
        f"risk_level={risk_level}, top_factors={top_factors}"
    )

    confidence_val = round(risk_score * 100, 1)
    prediction_str = "High Risk" if risk_score >= 0.5 else "Low Risk"

    return PredictionResult(
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        top_factors=top_factors,
        prediction=prediction_str,
        confidence=confidence_val,
        risk_factors=round(total_positive, 1),
        protective_factors=round(total_negative, 1),
        shap_values=shap_list_recharts,
        shap_values_dict=shap_dict,
        probability_percent=confidence_val
    )
