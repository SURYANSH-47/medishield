"""
shap_explain.py
===============
Explainable AI (XAI) module for the MediShield AI diabetes risk predictor.

Why Explainability Matters in Healthcare
-----------------------------------------
A black-box AI model that says "70% diabetes risk" is not enough for a doctor
or a patient to trust and act on. Clinicians need to know:
  ✦ Which features pushed the prediction toward "diabetic"?
  ✦ Which features pulled it toward "healthy"?
  ✦ Is the model using medically sensible signals?

SHAP (SHapley Additive exPlanations) answers these questions rigorously.

What is SHAP?
-------------
SHAP is a game-theory–based method for explaining the output of any machine
learning model. It assigns each feature an *importance value* (SHAP value)
for a particular prediction.

  • A POSITIVE SHAP value → feature pushed prediction TOWARD diabetes risk.
  • A NEGATIVE SHAP value → feature pushed prediction AWAY from diabetes risk.
  • SHAP values are additive:  base_value + sum(shap_values) = model output

How SHAP Values Are Interpreted
--------------------------------
Given a prediction of 0.73 (73% diabetes risk):
  • Base value (average model output)   :  0.38
  • Glucose         SHAP = +0.21  → pushed risk UP   (high glucose is bad)
  • BMI             SHAP = +0.09  → pushed risk UP   (high BMI is bad)
  • Age             SHAP = +0.07  → pushed risk UP   (older patient)
  • BloodPressure   SHAP = -0.02  → pushed risk DOWN (normal BP is good)

Pipeline in This Script
------------------------
  1. Load preprocessed tensors from preprocess.py
  2. Load trained DiabetesClassifier from saved_model/global_model.pth
  3. Build a SHAP-compatible model wrapper (handles sigmoid + tensor→numpy)
  4. Create a SHAP KernelExplainer using the training set as background
  5. Compute SHAP values on the test set
  6. Explain individual patients (top factors, risk level, direction)
  7. Display global feature importance (average |SHAP| across all patients)
  8. Generate and save publication-quality plots:
       - SHAP Summary Plot   (shap_plots/summary_plot.png)
       - Feature Importance  (shap_plots/feature_importance.png)
       - Waterfall Plot      (shap_plots/patient_<N>_waterfall.png)
  9. Support custom patient input for real-time explanation
"""

import os
import sys
import warnings
import numpy as np
import torch

# Suppress minor warnings for a clean hackathon demo
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import our own MediShield modules
# ---------------------------------------------------------------------------
from preprocess import preprocess          # full preprocessing pipeline
from model import DiabetesRiskPredictor    # the DNN architecture

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to saved model weights (created by train.py)
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "global_model.pth")

# Directory to save SHAP plots (created automatically if missing)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "shap_plots")

# Threshold for converting probability → binary prediction
THRESHOLD = 0.5

# How many test patients to explain individually
N_INDIVIDUAL_EXPLANATIONS = 3

# Feature names (must match preprocess.py column order)
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

# Healthcare units for each feature (for display only)
FEATURE_UNITS = {
    "Pregnancies":             "count",
    "Glucose":                 "mg/dL",
    "BloodPressure":           "mmHg",
    "SkinThickness":           "mm",
    "Insulin":                 "μU/mL",
    "BMI":                     "kg/m²",
    "DiabetesPedigreeFunction":"score",
    "Age":                     "years",
}


# ---------------------------------------------------------------------------
# STEP 1 — SHAP-compatible model wrapper
# ---------------------------------------------------------------------------

class SHAPModelWrapper:
    """
    Wraps the PyTorch DiabetesClassifier so SHAP can call it like a plain
    numpy function.

    SHAP's KernelExplainer expects a function  f(X) → probabilities  where:
      • X is a 2-D numpy array of shape (n_samples, n_features)
      • output is a 1-D numpy array of probabilities in [0, 1]

    The wrapper:
      1. Converts the numpy array to a float32 PyTorch tensor
      2. Runs the model in inference mode (no gradients, no dropout)
      3. Applies sigmoid to convert raw logits → probabilities
      4. Returns a numpy array
    """

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model  = model
        self.device = device
        # Ensure model is in evaluation mode
        self.model.eval()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)

        Returns
        -------
        probs : np.ndarray  shape (n_samples,)  values in [0, 1]
        """
        # Convert numpy → PyTorch tensor (float32 required by PyTorch layers)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

        # No gradient tracking needed during SHAP inference
        with torch.no_grad():
            # Forward pass → probabilities in [0, 1] (model already applies sigmoid)
            # Output shape: (n_samples, 1)
            probs = self.model(X_tensor)

            # Remove singleton dimension: (n_samples, 1) → (n_samples,)
            probs = probs.squeeze(1)

        # Return as numpy (CPU) for SHAP compatibility
        return probs.cpu().numpy()


# ---------------------------------------------------------------------------
# STEP 2 — Load model and data
# ---------------------------------------------------------------------------

def load_model_and_data() -> tuple:
    """
    Load the trained DiabetesClassifier and preprocessed tensors.

    Returns
    -------
    model   : DiabetesClassifier (eval mode, on device)
    data    : dict from preprocess() — X_train, X_test, y_train, y_test, etc.
    device  : torch.device (cpu or cuda)
    wrapper : SHAPModelWrapper
    """
    print("=" * 65)
    print("  STEP 1 — Loading Model & Data")
    print("=" * 65)

    # -- Device selection (GPU if CUDA available, else CPU) ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # -- Preprocessing pipeline -----------------------------------------------
    print("\n  Running preprocessing pipeline...")
    data = preprocess()

    print(f"\n  Tensors loaded:")
    print(f"    X_train : {tuple(data['X_train'].shape)}")
    print(f"    X_test  : {tuple(data['X_test'].shape)}")
    print(f"    y_train : {tuple(data['y_train'].shape)}")
    print(f"    y_test  : {tuple(data['y_test'].shape)}")

    # -- Model initialization -------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n  [ERROR] No trained model found at:\n    {MODEL_PATH}\n"
            f"  Please run:  python train.py\n  first to train and save the model."
        )

    n_features = data["X_train"].shape[1]   # should be 8
    model = DiabetesRiskPredictor(input_size=n_features).to(device)

    # Load saved weights (map_location handles CPU→GPU or GPU→CPU moves)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()   # critical: disables Dropout + uses BatchNorm running stats

    print(f"\n  Model loaded from  : {MODEL_PATH}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params   : {n_params:,}")

    # -- SHAP wrapper ---------------------------------------------------------
    wrapper = SHAPModelWrapper(model, device)

    return model, data, device, wrapper


# ---------------------------------------------------------------------------
# STEP 3 — Build SHAP explainer
# ---------------------------------------------------------------------------

def build_explainer(wrapper: SHAPModelWrapper, X_train: np.ndarray):
    """
    Create a SHAP KernelExplainer using the training set as background data.

    Why KernelExplainer?
    --------------------
    • Works with ANY model (no need to open PyTorch internals).
    • Uses a kernel trick to approximate Shapley values.
    • Slightly slower than DeepExplainer / GradientExplainer, but more
      stable and model-agnostic — ideal for hackathon demos.

    Background Data
    ---------------
    SHAP uses the background dataset to represent the "baseline" (what happens
    when a feature is masked / unknown). We use the mean of the training set
    as a compact 1-row background to keep computation fast.

    Parameters
    ----------
    wrapper  : SHAPModelWrapper  – callable f(X_numpy) → probabilities
    X_train  : np.ndarray        – training features (n_train, 8)

    Returns
    -------
    explainer : shap.KernelExplainer
    """
    import shap

    print("\n" + "=" * 65)
    print("  STEP 2 — Building SHAP KernelExplainer")
    print("=" * 65)

    # Use the mean of the training set as the single background sample.
    # A single-row background drastically speeds up KernelExplainer while
    # still capturing the "average" baseline behaviour.
    background = X_train.mean(axis=0, keepdims=True)   # shape (1, 8)

    print(f"\n  Background data shape  : {background.shape}")
    print(f"  Background (avg patient): {np.round(background[0], 2)}")

    # Create the explainer — wrapper.predict is the model function
    explainer = shap.KernelExplainer(wrapper.predict, background)

    print("\n  [OK] SHAP KernelExplainer ready.")
    return explainer


# ---------------------------------------------------------------------------
# STEP 4 — Compute SHAP values
# ---------------------------------------------------------------------------

def compute_shap_values(explainer, X_test: np.ndarray) -> np.ndarray:
    """
    Compute SHAP values for the test set.

    Parameters
    ----------
    explainer : shap.KernelExplainer
    X_test    : np.ndarray  shape (n_test, 8)

    Returns
    -------
    shap_values : np.ndarray  shape (n_test, 8)
        Each row is the SHAP decomposition for one patient.
        Each column corresponds to one feature.
    """
    import shap

    print("\n" + "=" * 65)
    print("  STEP 3 — Computing SHAP Values")
    print("=" * 65)
    print(f"\n  Explaining {len(X_test)} test patients...")
    print("  (This may take ~30–60 seconds for KernelExplainer)\n")

    # nsamples='auto' lets SHAP choose a good number of perturbations
    shap_values = explainer.shap_values(X_test, nsamples="auto")

    print(f"\n  [OK] SHAP values computed.")
    print(f"  SHAP values shape : {shap_values.shape}")
    return shap_values


# ---------------------------------------------------------------------------
# STEP 5 — Explain an individual patient
# ---------------------------------------------------------------------------

def explain_patient(
    patient_idx: int,
    X_test: np.ndarray,
    y_test: np.ndarray,
    shap_values: np.ndarray,
    wrapper: SHAPModelWrapper,
    scaler,
    label: str = None,
) -> None:
    """
    Print a detailed SHAP explanation for one patient.

    Displays:
      • Risk level and probability
      • Ground truth label
      • Feature values (original un-scaled)
      • Top 3 factors INCREASING risk (positive SHAP)
      • Top 3 factors DECREASING risk (negative SHAP)

    Parameters
    ----------
    patient_idx : int              – index into X_test / shap_values
    X_test      : np.ndarray       – scaled test features (n_test, 8)
    y_test      : np.ndarray       – ground truth labels
    shap_values : np.ndarray       – SHAP values (n_test, 8)
    wrapper     : SHAPModelWrapper – to get prediction probability
    scaler      : StandardScaler   – to inverse-transform features for display
    label       : str              – optional custom label for this patient
    """
    patient_x     = X_test[patient_idx : patient_idx + 1]   # shape (1, 8)
    patient_shap  = shap_values[patient_idx]                 # shape (8,)
    patient_prob  = wrapper.predict(patient_x)[0]            # scalar
    true_label    = int(y_test[patient_idx])

    # Inverse-transform to get interpretable original feature values
    original_x = scaler.inverse_transform(patient_x)[0]     # shape (8,)

    # Risk classification
    risk_level = "HIGH RISK" if patient_prob >= THRESHOLD else "LOW RISK"
    risk_emoji = "[HIGH]" if patient_prob >= THRESHOLD else "[LOW]"
    true_str   = "Diabetic" if true_label == 1 else "Healthy"

    header = label if label else f"Patient #{patient_idx + 1}"

    print("\n" + "-" * 65)
    print(f"  {risk_emoji}  {header}")
    print("-" * 65)
    print(f"  Prediction    :  {risk_level}  ({patient_prob * 100:.1f}%)")
    print(f"  Ground Truth  :  {true_str}")
    correct = (patient_prob >= THRESHOLD) == bool(true_label)
    print(f"  Model correct :  {'[OK] YES' if correct else '[X] NO'}")

    # -- Feature values table -------------------------------------------------
    print(f"\n  Feature Values:")
    print(f"  {'Feature':<28}  {'Value':>10}  {'Unit':<12}  {'SHAP':>8}")
    print(f"  {'-' * 60}")
    for i, name in enumerate(FEATURE_NAMES):
        unit   = FEATURE_UNITS.get(name, "")
        val    = original_x[i]
        sv     = patient_shap[i]
        arrow  = "^" if sv > 0 else ("v" if sv < 0 else "-")
        print(f"  {name:<28}  {val:>10.2f}  {unit:<12}  {arrow}{abs(sv):>6.4f}")

    # -- Top risk-increasing features (positive SHAP) -------------------------
    positive_mask = patient_shap > 0
    if positive_mask.any():
        top_pos_idx = np.argsort(patient_shap)[::-1][:3]
        print(f"\n  🔺 Top Factors INCREASING Diabetes Risk:")
        for rank, idx in enumerate(top_pos_idx, 1):
            if patient_shap[idx] > 0:
                name = FEATURE_NAMES[idx]
                val  = original_x[idx]
                sv   = patient_shap[idx]
                unit = FEATURE_UNITS.get(name, "")
                print(f"     {rank}. {name:<28} = {val:.2f} {unit:<10} "
                      f"[+{sv:.4f} SHAP]")
    else:
        print(f"\n  🔺 No features significantly increased risk.")

    # -- Top risk-decreasing features (negative SHAP) -------------------------
    negative_mask = patient_shap < 0
    if negative_mask.any():
        top_neg_idx = np.argsort(patient_shap)[:3]
        print(f"\n  🔻 Top Factors DECREASING Diabetes Risk:")
        for rank, idx in enumerate(top_neg_idx, 1):
            if patient_shap[idx] < 0:
                name = FEATURE_NAMES[idx]
                val  = original_x[idx]
                sv   = patient_shap[idx]
                unit = FEATURE_UNITS.get(name, "")
                print(f"     {rank}. {name:<28} = {val:.2f} {unit:<10} "
                      f"[{sv:.4f} SHAP]")
    else:
        print(f"\n  🔻 No features significantly decreased risk.")

    print("-" * 65)


# ---------------------------------------------------------------------------
# STEP 6 — Global feature importance
# ---------------------------------------------------------------------------

def display_global_importance(shap_values: np.ndarray) -> None:
    """
    Display global feature importance as mean absolute SHAP values.

    Mean |SHAP| tells us which features matter most ON AVERAGE across all
    patients — not just for one individual.

    Parameters
    ----------
    shap_values : np.ndarray  shape (n_test, 8)
    """
    # Mean absolute SHAP value per feature (averaged over all test patients)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)   # shape (8,)

    # Sort features by importance (highest first)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    print("\n" + "=" * 65)
    print("  GLOBAL FEATURE IMPORTANCE  (mean |SHAP| across all patients)")
    print("=" * 65)
    print(f"\n  {'Rank':<6}  {'Feature':<28}  {'Mean |SHAP|':>12}  {'Bar'}")
    print(f"  {'-' * 60}")

    max_val = mean_abs_shap.max()

    for rank, idx in enumerate(sorted_idx, 1):
        name     = FEATURE_NAMES[idx]
        val      = mean_abs_shap[idx]
        bar_len  = int((val / max_val) * 30)
        bar      = "#" * bar_len + "." * (30 - bar_len)
        print(f"  {rank:<6}  {name:<28}  {val:>12.4f}  {bar}")

    print(f"\n  Interpretation:")
    top_name = FEATURE_NAMES[sorted_idx[0]]
    print(f"  • '{top_name}' is the most influential feature overall.")
    print(f"  • Features at the top drive the model's decisions most strongly.")
    print(f"  • Features at the bottom have little impact on predictions.")
    print("=" * 65)


# ---------------------------------------------------------------------------
# STEP 7 — Save SHAP plots
# ---------------------------------------------------------------------------

def save_shap_plots(
    explainer,
    shap_values: np.ndarray,
    X_test: np.ndarray,
    scaler,
) -> None:
    """
    Generate and save publication-quality SHAP visualizations.

    Plots saved:
      1. summary_plot.png    — beeswarm: feature impact vs feature value
      2. feature_importance.png — bar chart of mean |SHAP|
      3. patient_<N>_waterfall.png — individual waterfall plots

    Parameters
    ----------
    explainer   : shap.KernelExplainer
    shap_values : np.ndarray  shape (n_test, 8)
    X_test      : np.ndarray  shape (n_test, 8) — scaled features
    scaler      : StandardScaler — for display in original scale
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe for all platforms
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs(PLOT_DIR, exist_ok=True)
    print("\n" + "=" * 65)
    print("  STEP 4 — Generating SHAP Plots")
    print("=" * 65)

    # Inverse-transform features to original scale for readable plots
    X_original = scaler.inverse_transform(X_test)

    # -- Plot 1: SHAP Summary Plot (beeswarm) --------------------------------
    # Each dot is one patient.
    # X-axis = SHAP value (impact on prediction)
    # Color  = feature value (red = high, blue = low)
    # Y-axis = features sorted by total impact
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_original,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_type="dot",
    )
    plt.title(
        "MediShield AI — SHAP Summary Plot\nFeature Impact on Diabetes Risk",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    summary_path = os.path.join(PLOT_DIR, "summary_plot.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [SAVED] Summary plot      → {summary_path}")

    # -- Plot 2: Feature Importance Bar Chart --------------------------------
    # Mean absolute SHAP value per feature — overall importance ranking.
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_original,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_type="bar",
    )
    plt.title(
        "MediShield AI — Feature Importance\nMean |SHAP Value| (Impact on Prediction)",
        fontsize=13, fontweight="bold", pad=15
    )
    plt.tight_layout()
    importance_path = os.path.join(PLOT_DIR, "feature_importance.png")
    plt.savefig(importance_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [SAVED] Feature importance → {importance_path}")

    # -- Plot 3: Waterfall plots for first N_INDIVIDUAL_EXPLANATIONS patients -
    # Waterfall shows how each feature additively contributes to move the
    # prediction from the base value up/down to the final output.
    for i in range(min(N_INDIVIDUAL_EXPLANATIONS, len(X_test))):
        try:
            # Build a shap.Explanation object for the waterfall API
            explanation = shap.Explanation(
                values     = shap_values[i],
                base_values= explainer.expected_value,
                data       = X_original[i],
                feature_names = FEATURE_NAMES,
            )
            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(explanation, show=False)
            plt.title(
                f"MediShield AI — Patient #{i+1} SHAP Waterfall\n"
                f"How features moved the prediction from baseline",
                fontsize=12, fontweight="bold", pad=12
            )
            plt.tight_layout()
            wf_path = os.path.join(PLOT_DIR, f"patient_{i+1}_waterfall.png")
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  [SAVED] Waterfall patient {i+1} → {wf_path}")
        except Exception as e:
            # Waterfall can occasionally fail on edge cases — skip gracefully
            print(f"  [SKIP]  Waterfall patient {i+1}: {e}")

    print(f"\n  All plots saved to: {PLOT_DIR}/")
    print("=" * 65)


# ---------------------------------------------------------------------------
# BONUS — Explain a custom patient
# ---------------------------------------------------------------------------

def explain_custom_patient(
    wrapper: SHAPModelWrapper,
    explainer,
    scaler,
    patient_values: dict = None,
) -> None:
    """
    Explain a single custom patient record (not from the dataset).

    This function is perfect for a live hackathon demo — doctors can enter
    real patient data and immediately see which features drive the prediction.

    Parameters
    ----------
    wrapper        : SHAPModelWrapper
    explainer      : shap.KernelExplainer
    scaler         : fitted StandardScaler from preprocess()
    patient_values : dict mapping feature name → raw value
                     If None, a realistic demo patient is used.

    Example
    -------
    explain_custom_patient(wrapper, explainer, scaler, {
        "Pregnancies": 6,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": 0,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    })
    """
    import shap

    print("\n" + "=" * 65)
    print("  BONUS — Custom Patient Explanation")
    print("=" * 65)

    # Use default demo patient if none provided (first row of Pima dataset)
    if patient_values is None:
        patient_values = {
            "Pregnancies":             6,
            "Glucose":                 148,
            "BloodPressure":           72,
            "SkinThickness":           35,
            "Insulin":                 0,
            "BMI":                     33.6,
            "DiabetesPedigreeFunction": 0.627,
            "Age":                     50,
        }
        print("\n  Using demo patient (Pima dataset row 1 — known diabetic):")

    # Build raw feature array in column order
    raw = np.array(
        [patient_values[f] for f in FEATURE_NAMES], dtype=np.float32
    ).reshape(1, -1)   # shape (1, 8)

    # Scale using the fitted StandardScaler (same transformation as training)
    scaled = scaler.transform(raw)   # shape (1, 8)

    # Prediction
    prob  = wrapper.predict(scaled)[0]
    level = "HIGH RISK [HIGH]" if prob >= THRESHOLD else "LOW RISK  [LOW]"

    # SHAP values for this patient
    sv = explainer.shap_values(scaled, nsamples="auto")   # shape (1, 8)
    sv = sv[0]                                             # shape (8,)

    print(f"\n  Patient Data:")
    for name in FEATURE_NAMES:
        unit = FEATURE_UNITS.get(name, "")
        print(f"    {name:<28} = {patient_values[name]:>8.2f}  {unit}")

    print(f"\n  Prediction: {level}  ({prob * 100:.1f}% diabetes probability)")
    print(f"  Base value (avg): {explainer.expected_value:.4f}")

    # Sort features by absolute SHAP impact
    sorted_idx = np.argsort(np.abs(sv))[::-1]

    print(f"\n  Top Contributing Factors:")
    print(f"  {'#':<4}  {'Feature':<28}  {'Value':>8}  {'SHAP':>10}  Impact")
    print(f"  {'-' * 60}")
    for rank, idx in enumerate(sorted_idx, 1):
        name   = FEATURE_NAMES[idx]
        val    = patient_values[name]
        unit   = FEATURE_UNITS.get(name, "")
        shap_v = sv[idx]
        impact = "^ Risk UP  " if shap_v > 0 else "v Risk DOWN"
        print(f"  {rank:<4}  {name:<28}  {val:>8.2f}  {shap_v:>+10.4f}  {impact}")

    # Save custom patient waterfall plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(PLOT_DIR, exist_ok=True)
        explanation = shap.Explanation(
            values      = sv,
            base_values = explainer.expected_value,
            data        = raw[0],
            feature_names = FEATURE_NAMES,
        )
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.title(
            "MediShield AI — Custom Patient SHAP Waterfall\n"
            f"Diabetes Risk: {prob * 100:.1f}%",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        custom_path = os.path.join(PLOT_DIR, "custom_patient_waterfall.png")
        plt.savefig(custom_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  [SAVED] Custom waterfall → {custom_path}")
    except Exception as e:
        print(f"\n  [SKIP]  Custom waterfall: {e}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def explain() -> dict:
    """
    Full SHAP explainability pipeline for MediShield AI.

    Returns
    -------
    dict with keys:
        'shap_values'   → np.ndarray (n_test, 8)
        'explainer'     → shap.KernelExplainer
        'wrapper'       → SHAPModelWrapper
        'data'          → preprocess() dict
    """

    # -- Ensure SHAP is available ---------------------------------------------
    try:
        import shap
        print(f"  [OK] SHAP version: {shap.__version__}")
    except ImportError:
        print("\n  [ERROR] SHAP is not installed.")
        print("  Install it with:  pip install shap")
        print("  Then re-run:      python shap_explain.py\n")
        sys.exit(1)

    # -- Ensure matplotlib is available ---------------------------------------
    try:
        import matplotlib
        print(f"  [OK] matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("\n  [WARNING] matplotlib not installed — plots will be skipped.")
        print("  Install it with:  pip install matplotlib\n")

    print("\n" + "=" * 65)
    print("  MediShield AI — SHAP Explainability Module")
    print("  Transparent AI for Trustworthy Healthcare")
    print("=" * 65)

    # -- Load model + data ----------------------------------------------------
    model, data, device, wrapper = load_model_and_data()

    # Extract numpy arrays (SHAP works with numpy, not PyTorch tensors)
    X_train_np = data["X_train"].numpy()          # shape (n_train, 8)
    X_test_np  = data["X_test"].numpy()           # shape (n_test, 8)
    y_test_np  = data["y_test"].numpy()           # shape (n_test,)
    scaler     = data["scaler"]                   # fitted StandardScaler

    # -- Build SHAP explainer -------------------------------------------------
    explainer = build_explainer(wrapper, X_train_np)

    # -- Compute SHAP values --------------------------------------------------
    shap_values = compute_shap_values(explainer, X_test_np)

    # -- Global feature importance --------------------------------------------
    display_global_importance(shap_values)

    # -- Individual patient explanations --------------------------------------
    print("\n" + "=" * 65)
    print(f"  INDIVIDUAL PATIENT EXPLANATIONS  (first {N_INDIVIDUAL_EXPLANATIONS})")
    print("=" * 65)

    for i in range(min(N_INDIVIDUAL_EXPLANATIONS, len(X_test_np))):
        explain_patient(
            patient_idx = i,
            X_test      = X_test_np,
            y_test      = y_test_np,
            shap_values = shap_values,
            wrapper     = wrapper,
            scaler      = scaler,
        )

    # -- SHAP plots -----------------------------------------------------------
    try:
        save_shap_plots(explainer, shap_values, X_test_np, scaler)
    except Exception as e:
        print(f"\n  [WARNING] Plot generation failed: {e}")
        print("  Install matplotlib:  pip install matplotlib")

    # -- Custom patient demo --------------------------------------------------
    explain_custom_patient(wrapper, explainer, scaler)

    # -- Final banner ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("  SHAP ANALYSIS COMPLETE")
    print("=" * 65)
    print("\n  What was generated:")
    print(f"    • Global feature importance ranking")
    print(f"    • Individual explanations for {min(N_INDIVIDUAL_EXPLANATIONS, len(X_test_np))} patients")
    print(f"    • Custom patient demo explanation")
    print(f"    • Plots saved to: {PLOT_DIR}/")
    print("\n  Key Takeaway:")
    print("    SHAP makes every prediction transparent — doctors and patients")
    print("    can see exactly which health factors drove each decision.")
    print("    This builds TRUST and supports INFORMED clinical decisions.")
    print("=" * 65 + "\n")

    return {
        "shap_values": shap_values,
        "explainer":   explainer,
        "wrapper":     wrapper,
        "data":        data,
    }


# ---------------------------------------------------------------------------
# Entry point — run directly:  python shap_explain.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = explain()
