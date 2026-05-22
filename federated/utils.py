"""
utils.py
========
Reusable utility functions shared across the federated learning module.

Responsibilities
----------------
1. load_hospital_data    — load a hospital CSV and return PyTorch tensors
2. get_model_weights     — extract model parameters as numpy arrays
3. set_model_weights     — load numpy arrays back into a model
4. compute_accuracy      — evaluate binary classification accuracy
5. save_status           — write federated_status.json for dashboard
6. load_status           — read the current status JSON
7. save_local_model      — persist a hospital's local DNN weights
8. save_global_model     — persist the aggregated global DNN weights
9. export_global_to_backend — copy global model into ml/saved_model/ so the
                              FastAPI backend can serve predictions from it

Design notes
------------
* Zero-imputation and StandardScaler are applied per hospital so each
  hospital's scaler is fitted only on its own data — this is the correct
  federated approach (no global statistics are shared).
* JSON status file is written atomically (write to temp then rename) so the
  frontend never reads a partially-written file.
* Local hospital weights are saved after every round so you can audit each
  hospital's training progress independently of the aggregated model.
"""

import json
import os
import sys
import tempfile
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hospital CSV files (produced by split_data.py)
DATA_DIR = os.path.join(PROJECT_ROOT, "federated", "data")

# JSON status file (consumed by dashboard frontend)
STATUS_DIR  = os.path.join(PROJECT_ROOT, "federated", "status")
STATUS_PATH = os.path.join(STATUS_DIR, "federated_status.json")

# Saved-model directory (local per-hospital + aggregated global)
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "federated", "saved_models")
GLOBAL_MODEL_PATH = os.path.join(SAVED_MODELS_DIR, "global_model.pth")

# Backend's expected model location — the FastAPI /predict endpoint
# loads from here, so the final aggregated global model is copied to it.
BACKEND_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml", "saved_model", "global_model.pth")

# Columns where 0 is physiologically impossible — impute with column median
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Target column name
TARGET_COL = "Outcome"

# Map hospital short-name → CSV filename
HOSPITAL_CSV = {
    "hospital_a": os.path.join(DATA_DIR, "hospital_a.csv"),
    "hospital_b": os.path.join(DATA_DIR, "hospital_b.csv"),
    "hospital_c": os.path.join(DATA_DIR, "hospital_c.csv"),
}

# Per-hospital metadata — used for both training and dashboard display
HOSPITAL_INFO = {
    "hospital_a": {
        "display_name": "Hospital A — Urban Wellness Clinic",
        "short_name":   "Hospital A",
        "specialty":    "Preventive Care",
    },
    "hospital_b": {
        "display_name": "Hospital B — Senior Care Hospital",
        "short_name":   "Hospital B",
        "specialty":    "Geriatric Medicine",
    },
    "hospital_c": {
        "display_name": "Hospital C — Metro General Hospital",
        "short_name":   "Hospital C",
        "specialty":    "General Medicine",
    },
}

# Back-compat shorthand (used by older callers — points at short_name)
HOSPITAL_DISPLAY = {key: info["short_name"] for key, info in HOSPITAL_INFO.items()}


# Local model file path per hospital
def local_model_path(hospital_name: str) -> str:
    """Return the on-disk path where this hospital's local weights are stored."""
    return os.path.join(SAVED_MODELS_DIR, f"{hospital_name}_local.pth")


# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

def load_hospital_data(
    hospital_name: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    """
    Load a hospital's private CSV partition and return train/test tensors.

    Steps:
      1. Read the hospital CSV
      2. Impute physiologically impossible zero values with column median
      3. Split into 80 % train / 20 % test (stratified)
      4. Fit StandardScaler on training data only (no data leakage)
      5. Convert to float32 PyTorch tensors

    Parameters
    ----------
    hospital_name : one of 'hospital_a', 'hospital_b', 'hospital_c'
    test_size     : fraction of data reserved for local evaluation
    random_state  : reproducibility seed

    Returns
    -------
    X_train, X_test, y_train, y_test : torch.Tensor
    scaler                           : fitted StandardScaler
    """
    from sklearn.model_selection import train_test_split  # local import avoids top-level dep

    csv_path = HOSPITAL_CSV.get(hospital_name)
    if csv_path is None:
        raise ValueError(
            f"Unknown hospital '{hospital_name}'. "
            f"Valid names: {list(HOSPITAL_CSV.keys())}"
        )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Hospital data not found at {csv_path}. "
            "Run split_data.py first."
        )

    # 1. Load
    df = pd.read_csv(csv_path)

    # 2. Impute invalid zeros with per-column median (ignoring zeros)
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            valid_median = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, valid_median)

    # 3. Feature / target split
    X = df.drop(columns=[TARGET_COL]).values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

    # 4. Stratified train-test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 5. Fit scaler on training data only — never on test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 6. Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    return X_train_t, X_test_t, y_train_t, y_test_t, scaler


# ---------------------------------------------------------------------------
# 2. Model weight helpers
# ---------------------------------------------------------------------------

def get_model_weights(model: torch.nn.Module) -> List[np.ndarray]:
    """
    Extract all trainable parameter tensors from a model as numpy arrays.

    This is what Flower transmits between client and server — only the
    weight values, never the raw data.
    """
    return [param.data.cpu().numpy() for param in model.parameters()]


def set_model_weights(model: torch.nn.Module, weights: List[np.ndarray]) -> None:
    """
    Load a list of numpy arrays back into the model's parameters in-place.

    The lengths must match the model's parameter list exactly.
    """
    params = list(model.parameters())
    if len(params) != len(weights):
        raise ValueError(
            f"Weight count mismatch: model has {len(params)} parameter tensors "
            f"but received {len(weights)}."
        )
    for param, weight in zip(params, weights):
        param.data = torch.tensor(weight, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. Accuracy computation
# ---------------------------------------------------------------------------

def compute_accuracy(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Compute binary classification accuracy on (X, y).

    Parameters
    ----------
    model     : DiabetesRiskPredictor (forward() already applies sigmoid)
    X         : feature tensor  (N, 8)
    y         : label tensor    (N,)
    threshold : probability threshold for positive prediction

    Returns
    -------
    accuracy as a float in [0, 1]
    """
    model.eval()
    with torch.no_grad():
        probs = model(X).squeeze()           # (N,) — sigmoid already applied inside forward()
        preds = (probs >= threshold).float() # (N,) binary predictions
        correct = (preds == y).sum().item()
    return correct / len(y)


# ---------------------------------------------------------------------------
# 4. JSON status file (dashboard integration)
# ---------------------------------------------------------------------------

def save_status(status: dict) -> None:
    """
    Write the federated learning status dict to JSON.

    The write is atomic: we write to a temp file in the same directory,
    then rename it to overwrite the real file. This prevents the frontend
    from reading a half-written JSON.

    Parameters
    ----------
    status : dict conforming to the federated_status.json schema
    """
    os.makedirs(STATUS_DIR, exist_ok=True)

    # Always stamp with current time
    status["timestamp"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    # Write atomically
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATUS_DIR, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(status, f, indent=2)
        os.replace(tmp_path, STATUS_PATH)   # atomic rename
    except Exception:
        os.unlink(tmp_path)
        raise


def load_status() -> dict:
    """
    Read the current federated_status.json file.

    Returns an empty dict if the file does not exist yet.
    """
    if not os.path.exists(STATUS_PATH):
        return {}
    with open(STATUS_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 5. Default status template
# ---------------------------------------------------------------------------

def make_initial_status(total_rounds: int, hospital_names: list) -> dict:
    """
    Build an initial status dict before training starts.
    Useful for the dashboard to show 'waiting' state immediately.
    """
    hospitals = []
    for name in hospital_names:
        info = HOSPITAL_INFO.get(name, {})
        hospitals.append({
            "name": info.get("short_name", name),
            "display_name": info.get("display_name", name),
            "specialty": info.get("specialty", ""),
            "local_accuracy": 0.0,
            "patients_count": 0,
            "training_status": "idle",
            "sync_status": "pending",
        })

    return {
        "current_round": 0,
        "total_rounds": total_rounds,
        "global_accuracy": 0.0,
        "training_status": "starting",
        "aggregation_status": "waiting",
        "aggregation_method": "FedAvg (simple average)",
        "privacy_guarantee": "Patient data never leaves hospitals — only model weights are shared.",
        "connected_hospitals": hospitals,
        "round_history": [],
    }


# ---------------------------------------------------------------------------
# 6. Model persistence — local + global checkpoints
# ---------------------------------------------------------------------------

def save_local_model(hospital_name: str, model: torch.nn.Module) -> str:
    """
    Persist a single hospital's local DNN weights to disk.

    Called by HospitalClient.fit() after each round so each hospital has
    an audit trail of its own model evolution.  These files are NEVER
    transmitted — they live on the hospital's own filesystem.

    Returns the absolute file path written.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    path = local_model_path(hospital_name)
    torch.save(model.state_dict(), path)
    return path


def save_global_model_weights(weights: List[np.ndarray], reference_model: torch.nn.Module) -> str:
    """
    Persist the aggregated global model weights to disk.

    `weights` is the FedAvg-aggregated list of numpy arrays.
    `reference_model` provides the architecture/parameter ordering so we
    can pack the numpy arrays back into a PyTorch state_dict.

    Returns the absolute file path written.
    """
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    # Load the weights into the reference model, then save its state_dict
    set_model_weights(reference_model, weights)
    torch.save(reference_model.state_dict(), GLOBAL_MODEL_PATH)
    return GLOBAL_MODEL_PATH


def export_global_to_backend() -> str:
    """
    Copy the latest aggregated global model from federated/saved_models/
    into ml/saved_model/ so the FastAPI backend can serve real predictions
    using the federated-trained weights.

    Returns the destination path.
    """
    import shutil

    if not os.path.exists(GLOBAL_MODEL_PATH):
        raise FileNotFoundError(
            f"No global model exists at {GLOBAL_MODEL_PATH}. "
            "Run federated/server.py to produce one."
        )

    os.makedirs(os.path.dirname(BACKEND_MODEL_PATH), exist_ok=True)
    shutil.copy2(GLOBAL_MODEL_PATH, BACKEND_MODEL_PATH)
    return BACKEND_MODEL_PATH
