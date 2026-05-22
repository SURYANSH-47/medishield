"""
client.py
=========
Flower client implementation — simulates a single hospital node.

Each hospital:
  1. Holds its own private dataset partition (never shared).
  2. Initialises a local copy of DiabetesRiskPredictor.
  3. Receives the latest global model weights from the server.
  4. Trains locally for a few epochs on its private data.
  5. Sends ONLY the updated model weights back to the server.
  6. Raw patient data never leaves the hospital.

Privacy guarantee
-----------------
The `fit()` method returns model weight arrays, not data rows.
The `evaluate()` method returns scalar metrics (loss, accuracy), not data.

Usage (inside simulation)
--------------------------
    from federated.client import make_client
    fl.simulation.start_simulation(client_fn=make_client, ...)
"""

import os
import sys
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr as fl
from flwr.common import Context

# ---------------------------------------------------------------------------
# Add project root to sys.path so we can import ml/model.py without
# modifying that file or its package structure.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_DIR = os.path.join(_PROJECT_ROOT, "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model import DiabetesRiskPredictor   # ml/model.py — DO NOT modify
from utils import (                       # federated/utils.py
    load_hospital_data,
    get_model_weights,
    set_model_weights,
    compute_accuracy,
    save_local_model,
    HOSPITAL_DISPLAY,
    HOSPITAL_INFO,
)

# ---------------------------------------------------------------------------
# Logging setup — one logger per client, identified by hospital name
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Local training hyper-parameters
# ---------------------------------------------------------------------------
LOCAL_EPOCHS = 5       # number of epochs each hospital trains per round
BATCH_SIZE   = 16      # small batch — realistic for ~100-sample datasets
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4


# ---------------------------------------------------------------------------
# HospitalClient — the core Flower client class
# ---------------------------------------------------------------------------

class HospitalClient(fl.client.NumPyClient):
    """
    Simulates a hospital participating in federated learning.

    Parameters
    ----------
    hospital_name : str
        One of 'hospital_a', 'hospital_b', 'hospital_c'.
        Used to locate the right CSV partition and label log messages.
    """

    def __init__(self, hospital_name: str) -> None:
        self.hospital_name = hospital_name
        info = HOSPITAL_INFO.get(hospital_name, {})
        self.display_name = info.get("short_name", hospital_name)
        self.full_name    = info.get("display_name", hospital_name)
        self.specialty    = info.get("specialty", "General")
        self.logger = logging.getLogger(self.display_name)

        # ------------------------------------------------------------------
        # Load this hospital's private dataset once at startup.
        # X_train / X_test are scaled float32 tensors.
        # y_train / y_test are float32 label tensors.
        # ------------------------------------------------------------------
        self.logger.info(f"Identity   : {self.full_name}")
        self.logger.info(f"Specialty  : {self.specialty}")
        self.logger.info("Loading PRIVATE local dataset (never transmitted)...")
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.scaler,
        ) = load_hospital_data(hospital_name)

        self.num_train = len(self.X_train)
        self.num_test  = len(self.X_test)
        self.logger.info(
            f"Local dataset ready — {self.num_train} train / "
            f"{self.num_test} test samples (all data stays on premises)."
        )

        # ------------------------------------------------------------------
        # Initialise a local DiabetesRiskPredictor.
        # Each round the server overwrites these weights with the global model.
        # ------------------------------------------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DiabetesRiskPredictor(input_size=8).to(self.device)

        # Build a DataLoader for the training split
        train_ds = TensorDataset(
            self.X_train.to(self.device),
            self.y_train.unsqueeze(1).to(self.device),   # (N, 1) for BCELoss
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

    # ------------------------------------------------------------------
    # get_parameters  — called by the server to initialise the global model
    # ------------------------------------------------------------------
    def get_parameters(self, config: dict) -> list:
        """Return current model weights as a list of numpy arrays."""
        return get_model_weights(self.model)

    # ------------------------------------------------------------------
    # fit  — local training round
    # ------------------------------------------------------------------
    def fit(self, parameters: list, config: dict) -> tuple:
        """
        Receive global weights, train locally, return updated weights.

        Parameters
        ----------
        parameters : global model weights (list of numpy arrays from server)
        config     : server-side config dict; may include 'local_epochs'

        Returns
        -------
        updated_weights : list of numpy arrays  (model weights only, NOT data)
        num_examples    : number of training samples used
        metrics         : dict with 'train_loss' and 'local_accuracy'
        """
        # 1. Set the received global weights into our local model
        set_model_weights(self.model, parameters)

        # 2. Honour server config; fall back to module default
        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))

        # 3. Local training
        # BCELoss is used because DiabetesRiskPredictor.forward() already applies sigmoid.
        criterion = nn.BCELoss()
        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        self.model.train()
        total_loss = 0.0
        total_samples = 0

        self.logger.info(
            f"Starting local training — {local_epochs} epoch(s), "
            f"{self.num_train} samples."
        )

        for epoch in range(1, local_epochs + 1):
            epoch_loss = 0.0
            epoch_samples = 0

            for X_batch, y_batch in self.train_loader:
                optimiser.zero_grad()
                probs = self.model(X_batch)       # sigmoid already applied inside forward()
                loss  = criterion(probs, y_batch)
                loss.backward()
                optimiser.step()

                epoch_loss    += loss.item() * X_batch.size(0)
                epoch_samples += X_batch.size(0)

            avg_epoch_loss = epoch_loss / epoch_samples
            total_loss    += epoch_loss
            total_samples += epoch_samples

            # Log every epoch so training progress is visible
            self.logger.info(
                f"  Epoch {epoch}/{local_epochs} — loss: {avg_epoch_loss:.4f}"
            )

        avg_loss = total_loss / total_samples

        # 4. Compute local accuracy on the training set (using updated weights)
        local_acc = compute_accuracy(
            self.model,
            self.X_train.to(self.device),
            self.y_train.to(self.device),
        )

        self.logger.info(
            f"Local training complete — avg loss: {avg_loss:.4f}, "
            f"local accuracy: {local_acc * 100:.2f}%"
        )

        # 5. Save this hospital's local model to its own filesystem.
        #    These weights stay on the hospital — only the in-memory copy is
        #    transmitted to the aggregation server for FedAvg.
        local_path = save_local_model(self.hospital_name, self.model)
        self.logger.info(f"Local model checkpoint saved → {local_path}")
        self.logger.info(
            "Transmitting model WEIGHTS ONLY to aggregation server "
            "(no patient data leaves this hospital)."
        )

        # 6. Return ONLY weights + metadata — never raw data
        return (
            get_model_weights(self.model),
            self.num_train,
            {
                "train_loss":     float(avg_loss),
                "local_accuracy": float(local_acc),
                "hospital_name":  self.hospital_name,
                "display_name":   self.display_name,
                "full_name":      self.full_name,
                "specialty":      self.specialty,
                "patients_count": self.num_train,
            },
        )

    # ------------------------------------------------------------------
    # evaluate  — test the global model on this hospital's local test set
    # ------------------------------------------------------------------
    def evaluate(self, parameters: list, config: dict) -> tuple:
        """
        Evaluate the global model weights on local test data.

        Returns
        -------
        loss     : float — cross-entropy loss on test set
        num_examples : int — number of test samples
        metrics  : dict with 'accuracy'
        """
        # Load global weights for evaluation
        set_model_weights(self.model, parameters)

        # BCELoss — model already outputs probabilities via sigmoid in forward()
        criterion = nn.BCELoss()
        self.model.eval()

        X_test = self.X_test.to(self.device)
        y_test = self.y_test.unsqueeze(1).to(self.device)  # (N, 1)

        with torch.no_grad():
            probs = self.model(X_test)            # sigmoid already applied inside forward()
            loss  = criterion(probs, y_test).item()

        acc = compute_accuracy(
            self.model,
            self.X_test.to(self.device),
            self.y_test.to(self.device),
        )

        self.logger.info(
            f"Evaluation — loss: {loss:.4f}, accuracy: {acc * 100:.2f}%"
        )

        return (
            float(loss),
            self.num_test,
            {
                "accuracy": float(acc),
                "hospital_name": self.hospital_name,
                "display_name": self.display_name,
            },
        )


# ---------------------------------------------------------------------------
# Factory function — required by fl.simulation.start_simulation()
# ---------------------------------------------------------------------------

# Map numeric client IDs (0, 1, 2) to hospital names
_CLIENT_ID_MAP = {
    0: "hospital_a",
    1: "hospital_b",
    2: "hospital_c",
}


def make_client(context: Context) -> fl.client.Client:
    """
    Client factory function used by Flower's simulation engine (>= 1.5).

    Flower passes a Context object; we extract the partition-id from
    node_config to determine which hospital this client represents.

    Returns
    -------
    fl.client.Client  (NumPyClient wrapped via .to_client())
    """
    partition_id = int(context.node_config.get("partition-id", 0))
    hospital_name = _CLIENT_ID_MAP.get(partition_id % 3, "hospital_a")
    return HospitalClient(hospital_name).to_client()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Testing HospitalClient for Hospital A...")
    client = HospitalClient("hospital_a")

    # Simulate get_parameters
    weights = client.get_parameters(config={})
    print(f"Number of weight arrays: {len(weights)}")

    # Simulate a single fit round with the current (random) weights
    updated_weights, n_samples, metrics = client.fit(weights, config={"local_epochs": 2})
    print(f"Trained on {n_samples} samples.")
    print(f"Local accuracy: {metrics['local_accuracy'] * 100:.2f}%")
    print("HospitalClient self-test passed.")
