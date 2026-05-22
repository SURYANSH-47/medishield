"""
evaluate.py
===========
Comprehensive evaluation script for the MediShield AI diabetes risk prediction model.

This script loads a trained DiabetesRiskPredictor model and evaluates its
performance on the test dataset using standard healthcare AI metrics.

Evaluation Pipeline
-------------------
  1. Load preprocessed test tensors from preprocess.py
  2. Load trained model weights from saved_model/global_model.pth
  3. Initialize DiabetesRiskPredictor
  4. Run inference on test set (GPU support if available)
  5. Convert probabilities to binary predictions (threshold = 0.5)
  6. Calculate metrics:
     - Accuracy: Overall correctness
     - Precision: True positives / (True positives + False positives)
     - Recall: True positives / (True positives + False negatives)
     - F1-Score: Harmonic mean of precision and recall
     - ROC-AUC: Area under receiver operating characteristic curve
  7. Generate confusion matrix
  8. Display prediction examples

Healthcare Context
------------------
In diabetes prediction:
  - High RECALL is critical: missing diabetic patients (false negatives) is costly
  - High PRECISION is also important: false alarms can cause unnecessary anxiety
  - ROC-AUC summarizes performance across all thresholds
  - Confusion matrix shows exactly where the model makes mistakes
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Import our modules
# ---------------------------------------------------------------------------
from preprocess import preprocess
from model import DiabetesRiskPredictor

# scikit-learn for evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the saved model weights
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "global_model.pth")

# Decision threshold for binary classification
THRESHOLD = 0.5

# Number of example predictions to display
N_EXAMPLES = 5



# ---------------------------------------------------------------------------
# Helper function: Print a formatted metrics table
# ---------------------------------------------------------------------------

def print_metrics_table(
    accuracy: float,
    precision: float,
    recall: float,
    f1: float,
    roc_auc: float,
) -> None:
    """
    Pretty-print evaluation metrics in a formatted table.

    Parameters
    ----------
    accuracy : float  – overall correctness (0 to 1)
    precision : float – positive predictions that were correct (0 to 1)
    recall : float    – actual positives that were identified (0 to 1)
    f1 : float        – harmonic mean of precision and recall (0 to 1)
    roc_auc : float   – area under ROC curve (0 to 1)
    """
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    print(f"\n  {'Metric':<20}  {'Score':>10}  {'Description':<35}")
    print("  " + "-" * 68)
    print(
        f"  {'Accuracy':<20}  {accuracy:>10.4f}  "
        f"Overall correctness"
    )
    print(
        f"  {'Precision':<20}  {precision:>10.4f}  "
        f"Positive predictions that were correct"
    )
    print(
        f"  {'Recall':<20}  {recall:>10.4f}  "
        f"CRITICAL: Fraction of diabetics identified"
    )
    print(
        f"  {'F1-Score':<20}  {f1:>10.4f}  "
        f"Harmonic mean of precision & recall"
    )
    print(
        f"  {'ROC-AUC':<20}  {roc_auc:>10.4f}  "
        f"Performance across all thresholds"
    )
    print("  " + "-" * 68)


# ---------------------------------------------------------------------------
# Helper function: Print confusion matrix with healthcare interpretation
# ---------------------------------------------------------------------------

def print_confusion_matrix_healthcare(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Compute and display confusion matrix with healthcare-specific interpretation.

    Confusion Matrix layout:
    ┌───────────────────┬─────────────┬─────────────┐
    │                   │ Pred Neg    │ Pred Pos    │
    ├───────────────────┼─────────────┼─────────────┤
    │ True Negative (TN)│    TN       │    FP       │
    │ True Positive (TP)│    FN       │    TP       │
    └───────────────────┴─────────────┴─────────────┘

    Healthcare Interpretation:
    - TN: Correctly identified non-diabetic → GOOD
    - TP: Correctly identified diabetic → GOOD
    - FN: Missed diabetic (False Negative) → DANGEROUS, should be minimized
    - FP: False alarm (False Positive) → Unnecessary anxiety, but safer than FN

    Parameters
    ----------
    y_true : np.ndarray  – ground truth labels (0 or 1)
    y_pred : np.ndarray  – binary predictions (0 or 1)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\n" + "=" * 70)
    print("CONFUSION MATRIX & HEALTHCARE INTERPRETATION")
    print("=" * 70)

    print("\n  Confusion Matrix Layout:")
    print(f"  ┌─────────────────────────────┬──────────┬──────────┐")
    print(f"  │                             │ Pred Neg │ Pred Pos │")
    print(f"  ├─────────────────────────────┼──────────┼──────────┤")
    print(f"  │ Actual Negative (Healthy)   │   {tn:>4d} │   {fp:>4d} │")
    print(f"  │ Actual Positive (Diabetic)  │   {fn:>4d} │   {tp:>4d} │")
    print(f"  └─────────────────────────────┴──────────┴──────────┘")

    print("\n  Detailed Breakdown:")
    print(f"  [+] True Negatives (TN):  {tn:>4d}  — Correctly identified as healthy")
    print(f"  [+] True Positives (TP):  {tp:>4d}  — Correctly identified as diabetic")
    print(f"  [-] False Positives (FP): {fp:>4d}  — Healthy flagged as diabetic")
    print(f"  [-] False Negatives (FN): {fn:>4d}  — Diabetic missed (MOST DANGEROUS)")

    print("\n  Healthcare Implications:")
    if fn > 0:
        print(f"  ⚠️  WARNING: {fn} diabetic patients were missed (False Negatives)")
        print(f"      In healthcare, missing disease is more dangerous than false alarms")
    if fp > 0:
        print(f"  ℹ️  {fp} healthy patients flagged for follow-up (False Positives)")
        print(f"      These may cause unnecessary anxiety but are preferable to missing cases")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Helper function: Display example predictions
# ---------------------------------------------------------------------------

def display_example_predictions(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    probs: torch.Tensor,
    predictions: torch.Tensor,
    n_examples: int = 5,
) -> None:
    """
    Display example predictions to help understand model behavior.

    Parameters
    ----------
    X_test : torch.Tensor        – test features shape (N_test, 8)
    y_test : torch.Tensor        – ground truth labels shape (N_test,)
    probs : torch.Tensor         – predicted probabilities shape (N_test,)
    predictions : torch.Tensor   – binary predictions shape (N_test,)
    n_examples : int             – number of examples to show
    """
    print("\n" + "=" * 70)
    print("SAMPLE PATIENT PREDICTIONS")
    print("=" * 70)

    # Move tensors to CPU for display
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    probs_np = probs.cpu().numpy()
    predictions_np = predictions.cpu().numpy()

    feature_names = [
        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age",
    ]

    print(f"\nShowing {n_examples} example predictions:\n")

    for i in range(min(n_examples, len(X_test))):
        actual_label = "DIABETIC" if y_test_np[i] == 1 else "HEALTHY"
        pred_label = "DIABETIC" if predictions_np[i] == 1 else "HEALTHY"
        prob_percent = probs_np[i] * 100
        correct = "[OK]" if predictions_np[i] == y_test_np[i] else "[X]"

        print(f"  Patient {i + 1}:")
        print(f"    Actual Label  : {actual_label}")
        print(f"    Prediction    : {pred_label}")
        print(f"    Confidence    : {prob_percent:.2f}%  {correct}")
        print(f"    Features      : " + ", ".join(
            [f"{name}={val:.2f}" for name, val in zip(feature_names, X_test_np[i])]
        ))
        print()



# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    """
    Complete evaluation pipeline for the trained diabetes prediction model.

    Returns
    -------
    dict with keys: 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                    'confusion_matrix', 'y_true', 'y_pred', 'probs'
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Preprocess and load test data
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1 — Loading Test Data")
    print("=" * 70)

    # Run preprocessing pipeline to get test tensors
    data = preprocess()

    X_test = data["X_test"]
    y_test = data["y_test"]

    print(f"\n  Test set shapes:")
    print(f"    X_test: {tuple(X_test.shape)}")
    print(f"    y_test: {tuple(y_test.shape)}")

    # -----------------------------------------------------------------------
    # STEP 2 — Select device (GPU if available)
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Move test data to device
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # -----------------------------------------------------------------------
    # STEP 3 — Load trained model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2 — Loading Trained Model")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}.\n"
            f"Please run: python train.py"
        )

    # Initialize model
    model = DiabetesRiskPredictor(input_size=8).to(device)

    # Load pre-trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"[OK] Model loaded from: {MODEL_PATH}")

    # Print model architecture
    print(f"\n  Model Architecture:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # STEP 4 — Run inference on test set
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3 — Running Inference")
    print("=" * 70)

    # Set model to evaluation mode
    # - Disables Dropout
    # - Uses running statistics for BatchNorm (if present)
    model.eval()

    # Disable gradient computation to save memory and speed up inference
    with torch.no_grad():
        # Forward pass: model returns probabilities [0, 1] due to sigmoid
        # Output shape: (N_test, 1)
        logits = model(X_test)

        # squeeze(1) removes the singleton dimension: (N_test, 1) → (N_test,)
        probs = logits.squeeze(1)

        # Convert probabilities to binary predictions using threshold
        # predictions shape: (N_test,)
        predictions = (probs >= THRESHOLD).float()

    print(f"\n  Inference complete on {len(X_test)} test samples")
    print(f"    Probability range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(
        f"    Predictions: {int(predictions.sum())} positive, "
        f"{len(predictions) - int(predictions.sum())} negative"
    )

    # -----------------------------------------------------------------------
    # STEP 5 — Convert tensors to numpy for sklearn metrics
    # -----------------------------------------------------------------------
    y_true = y_test.cpu().numpy().astype(int)
    y_pred = predictions.cpu().numpy().astype(int)
    probs_np = probs.cpu().numpy()

    # -----------------------------------------------------------------------
    # STEP 6 — Calculate evaluation metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4 — Calculating Metrics")
    print("=" * 70)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs_np)

    print("\n  Metrics calculated:")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}  (CRITICAL in healthcare)")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    ROC-AUC   : {roc_auc:.4f}")

    # -----------------------------------------------------------------------
    # STEP 7 — Print metrics table
    # -----------------------------------------------------------------------
    print_metrics_table(accuracy, precision, recall, f1, roc_auc)

    # -----------------------------------------------------------------------
    # STEP 8 — Print confusion matrix with healthcare interpretation
    # -----------------------------------------------------------------------
    print_confusion_matrix_healthcare(y_true, y_pred)

    # -----------------------------------------------------------------------
    # STEP 9 — Print classification report
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 70)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Healthy", "Diabetic"],
            digits=4,
        )
    )

    # -----------------------------------------------------------------------
    # STEP 10 — Display example predictions
    # -----------------------------------------------------------------------
    display_example_predictions(X_test, y_test, probs, predictions, N_EXAMPLES)

    # -----------------------------------------------------------------------
    # STEP 11 — Healthcare insights
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HEALTHCARE INSIGHTS")
    print("=" * 70)

    print("\nKey Metrics Explained:")
    print("\n  1. RECALL (Most Important in Diabetes Screening)")
    print(f"     Current Recall: {recall:.4f}")
    print("     ↳ Of all diabetic patients, what fraction did we catch?")
    print("     ↳ Missing diabetics (false negatives) can lead to serious")
    print("       health complications")
    print(f"     ↳ With current recall={recall:.1%}, we missed ~{(1-recall)*100:.1f}% of cases")

    print("\n  2. PRECISION (Cost of False Alarms)")
    print(f"     Current Precision: {precision:.4f}")
    print("     ↳ Of patients we flagged as diabetic, how many really are?")
    print("     ↳ False positives cause unnecessary anxiety and follow-up tests")

    print("\n  3. ACCURACY (Overall Correctness)")
    print(f"     Current Accuracy: {accuracy:.4f}")
    print(f"     ↳ Overall, we're correct ~{accuracy*100:.1f}% of the time")

    print("\n  4. ROC-AUC (Performance Across All Thresholds)")
    print(f"     Current ROC-AUC: {roc_auc:.4f}")
    print("     ↳ How well does the model distinguish diabetic vs healthy?")
    print("     ↳ 0.5 = random guessing, 1.0 = perfect classification")

    print("\n" + "=" * 70)

    # -----------------------------------------------------------------------
    # Return results dictionary
    # -----------------------------------------------------------------------
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs_np,
        "model": model,
    }

    return results


# ---------------------------------------------------------------------------
# Entry point – run directly: python evaluate.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        results = evaluate()

        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"[OK] Model evaluation successful!")
        print(f"\nFinal Metrics:")
        print(f"  • Accuracy:  {results['accuracy']:.4f}")
        print(f"  • Precision: {results['precision']:.4f}")
        print(f"  • Recall:    {results['recall']:.4f}")
        print(f"  • F1-Score:  {results['f1']:.4f}")
        print(f"  • ROC-AUC:   {results['roc_auc']:.4f}")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure model has been trained: python train.py")
        print("  2. Check that saved_model/global_model.pth exists")
        print("  3. Verify preprocess.py is in the same directory")
        print("  4. Verify model.py is in the same directory")
