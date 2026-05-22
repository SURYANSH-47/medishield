"""
train.py
========
Training script for the MediShield AI diabetes risk classifier.

Pipeline
--------
  1. Run the full preprocessing pipeline  (preprocess.py)
  2. Load tensors: X_train, X_test, y_train, y_test
  3. Initialise the DiabetesClassifier DNN  (model.py)
  4. Define loss (BCELoss) and optimiser (Adam, lr=0.001)
  5. Train for 50 epochs – forward pass → loss → backprop → weight update
  6. Print epoch number, training loss, and training accuracy each epoch
  7. Evaluate on the test set and print final test accuracy
  8. Save the trained model to saved_model/global_model.pth

Binary classification:
  0 → no diabetes
  1 → diabetes risk

Predictions are thresholded at 0.5.
"""

import os
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import our own modules
# ---------------------------------------------------------------------------
# preprocess() runs the full data pipeline and returns a dict of tensors
from preprocess import preprocess

# DiabetesClassifier is the DNN defined in model.py
from model import DiabetesRiskPredictor

# ---------------------------------------------------------------------------
# Hyperparameters  (all gathered at the top for easy hackathon tweaking)
# ---------------------------------------------------------------------------
EPOCHS        = 50          # number of full passes through the training set
LEARNING_RATE = 0.001       # Adam learning rate
THRESHOLD     = 0.5         # decision boundary for binary predictions

# Where to save the final trained model
SAVE_DIR   = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH = os.path.join(SAVE_DIR, "global_model.pth")


# ---------------------------------------------------------------------------
# Helper – calculate accuracy given probabilities and true labels
# ---------------------------------------------------------------------------

def compute_accuracy(probs: torch.Tensor, labels: torch.Tensor, threshold: float = THRESHOLD) -> float:
    """
    Convert sigmoid probabilities to 0/1 predictions using `threshold`,
    compare against the ground-truth labels, and return accuracy as a float.

    Parameters
    ----------
    probs     : 1-D tensor of sigmoid probabilities (values in [0, 1])
    labels    : 1-D tensor of ground-truth labels   (0 or 1)
    threshold : decision boundary (default 0.5)

    Returns
    -------
    accuracy  : float in [0.0, 1.0]
    """
    # Apply threshold to get hard 0/1 predictions
    predictions = (probs >= threshold).float()

    # Count how many predictions match the ground truth
    correct = (predictions == labels).sum().item()

    # Divide by total number of samples to get accuracy
    accuracy = correct / labels.size(0)
    return accuracy


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train() -> dict:
    """
    Full training pipeline for the DiabetesClassifier.

    Steps
    -----
    1. Preprocess dataset → tensors
    2. Move tensors to GPU (if available)
    3. Initialise model, BCELoss, and Adam optimiser
    4. Train for EPOCHS epochs with full-batch updates
    5. Evaluate on test set
    6. Save model weights to disk

    Returns
    -------
    dict with keys 'train_loss', 'train_acc', 'test_acc', 'model'
    """

    # -----------------------------------------------------------------------
    # STEP 1 — Preprocess the raw CSV into PyTorch tensors
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("  STEP 1 — Data Preprocessing")
    print("=" * 60)

    # preprocess() returns a dict with keys:
    #   'X_train', 'X_test', 'y_train', 'y_test', 'scaler', 'feature_names'
    data = preprocess()

    # Unpack the four tensors we need for training and evaluation
    X_train = data["X_train"]   # shape: (N_train, 8)
    X_test  = data["X_test"]    # shape: (N_test,  8)
    y_train = data["y_train"]   # shape: (N_train,)   — 0 or 1 labels
    y_test  = data["y_test"]    # shape: (N_test,)    — 0 or 1 labels

    # Derive the number of input features from the training tensor
    n_features = X_train.shape[1]   # should be 8 for the Pima dataset

    print(f"\n  Tensor shapes loaded:")
    print(f"    X_train : {tuple(X_train.shape)}")
    print(f"    X_test  : {tuple(X_test.shape)}")
    print(f"    y_train : {tuple(y_train.shape)}")
    print(f"    y_test  : {tuple(y_test.shape)}")

    # -----------------------------------------------------------------------
    # BONUS — GPU support: use CUDA if a compatible GPU is available
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")
    if device.type == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # Move all tensors to the selected device
    X_train = X_train.to(device)
    X_test  = X_test.to(device)
    y_train = y_train.to(device)
    y_test  = y_test.to(device)

    # -----------------------------------------------------------------------
    # STEP 2 — Initialise the DNN model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 2 — Model Initialisation")
    print("=" * 60)

    # DiabetesClassifier: 8 → 64 → 32 → 16 → 1  (raw logit output)
    model = DiabetesRiskPredictor(input_size=n_features).to(device)

    print(f"\n  Architecture:")
    print(model)

    # Count trainable parameters
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f"\n  Trainable parameters: {trainable_params:,}")

    # -----------------------------------------------------------------------
    # STEP 3 — Define loss function and optimiser
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 3 — Loss & Optimiser")
    print("=" * 60)

    # BCELoss requires probabilities in [0, 1].
    # NOTE: DiabetesRiskPredictor already applies sigmoid in forward pass,
    # so model(X_train) returns probabilities directly (not raw logits).
    criterion = nn.BCELoss()

    # Adam adapts the learning rate per parameter — great for small datasets
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"\n  Loss function : nn.BCELoss()")
    print(f"  Optimiser     : Adam  (lr={LEARNING_RATE})")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Threshold     : {THRESHOLD}")

    # -----------------------------------------------------------------------
    # STEP 4 — Training loop (50 epochs)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 4 — Training")
    print("=" * 60)
    print(f"\n  {'Epoch':>6}  {'Train Loss':>12}  {'Train Acc':>12}")
    print("  " + "-" * 36)

    # Track metrics per epoch for optional downstream use
    history = {"train_loss": [], "train_acc": []}

    for epoch in range(1, EPOCHS + 1):

        # --- Put model in training mode (enables Dropout + BatchNorm stats) ---
        model.train()

        # --- Forward pass ---
        # model(X_train) returns probabilities [0, 1] of shape (N_train, 1)
        # squeeze(1) removes the dimension to get shape (N_train,) for BCELoss
        probs = model(X_train).squeeze(1)
        loss = criterion(probs, y_train)

        # --- Backpropagation ---
        # Zero out gradients from the previous step to avoid accumulation
        optimizer.zero_grad()

        # Compute gradients of loss w.r.t. all model parameters
        loss.backward()

        # --- Update weights ---
        # Adam uses the gradients to update each parameter
        optimizer.step()

        # --- Metrics for this epoch ---
        # Detach from the computation graph before any numpy/python operations
        with torch.no_grad():
            train_acc = compute_accuracy(probs.detach(), y_train)

        # Record history
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)

        # --- Clean progress log – print every epoch ---
        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  "
              f"Loss: {loss.item():>8.4f}  |  "
              f"Accuracy: {train_acc * 100:>6.2f}%")

    print("  " + "-" * 36)
    print(f"\n  Training complete!")
    print(f"  Final train loss : {history['train_loss'][-1]:.4f}")
    print(f"  Final train acc  : {history['train_acc'][-1] * 100:.2f}%")

    # -----------------------------------------------------------------------
    # STEP 5 — Evaluate on the test set
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 5 — Test Set Evaluation")
    print("=" * 60)

    # Switch to evaluation mode:
    # - Dropout layers are disabled (all neurons active)
    # - BatchNorm uses running statistics (not batch statistics)
    model.eval()

    with torch.no_grad():
        # Forward pass on the test set
        # model(X_test) already returns probabilities [0, 1] (sigmoid applied in forward pass)
        # squeeze(1) removes dimension: (N_test, 1) → (N_test,)
        test_probs = model(X_test).squeeze(1)

        # NOTE: Do NOT apply torch.sigmoid() again – model already includes it!
        # Applying sigmoid twice would be incorrect and cause poor loss values.

        # Compute test loss
        test_loss = criterion(test_probs, y_test).item()

        # Compute test accuracy with 0.5 threshold
        test_acc = compute_accuracy(test_probs, y_test)

    print(f"\n  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc * 100:.2f}%")

    # -----------------------------------------------------------------------
    # STEP 6 — Save the trained model
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEP 6 — Saving Model")
    print("=" * 60)

    # Create the output directory automatically if it does not exist
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Save the model's state dict (weights + biases)
    # To reload: model.load_state_dict(torch.load(MODEL_PATH))
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"\n  Model saved to : {MODEL_PATH}")

    # -----------------------------------------------------------------------
    # Return a summary dict (useful for evaluate.py or notebooks)
    # -----------------------------------------------------------------------
    history["test_loss"] = test_loss
    history["test_acc"]  = test_acc
    history["model"]     = model
    history["data"]      = data

    return history


# ---------------------------------------------------------------------------
# Entry point – run directly:  python train.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = train()

    # Final summary banner
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    print(f"  Epochs trained   : {EPOCHS}")
    print(f"  Final train acc  : {results['train_acc'][-1] * 100:.2f}%")
    print(f"  Final test acc   : {results['test_acc'] * 100:.2f}%")
    print(f"  Model saved to   : {MODEL_PATH}")
    print("=" * 60)
    print("\n  MediShield AI model is ready for deployment!\n")
