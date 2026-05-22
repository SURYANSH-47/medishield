"""
strategy.py
===========
Custom Federated Averaging (FedAvg) strategy for MediShield.

What is FedAvg?
---------------
After each training round, every hospital sends its updated model weights
to the central server. The server computes a weighted average of all
received weight arrays — weighted by the number of training examples each
hospital contributed. This produces a new "global" model that benefits
from all hospitals' data without any hospital sharing raw data.

Formula (for each weight tensor w):
    w_global = Σ (n_i / N_total) * w_i
    where n_i = samples from hospital i,  N_total = Σ n_i

MediShieldFedAvg extends Flower's built-in FedAvg strategy with:
  * Per-round logging (round number, participating hospitals, accuracy)
  * Writing federated_status.json after every round for dashboard use
  * Tracking round_history (list of {round, accuracy} dicts)

Configurable parameters
------------------------
  num_rounds : total rounds planned (used for status reporting only)
  min_fit_clients    : minimum clients that must train before aggregation
  min_evaluate_clients : minimum clients that must evaluate
  min_available_clients : minimum clients that must be reachable
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    EvaluateRes,
    FitIns,
    EvaluateIns,
)
from flwr.server.client_proxy import ClientProxy

import os
import sys

from utils import (   # federated/utils.py
    save_status,
    load_status,
    save_global_model_weights,
    HOSPITAL_DISPLAY,
    HOSPITAL_INFO,
)

# Allow importing the model architecture for global-model checkpoint saving
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ML_DIR       = os.path.join(_PROJECT_ROOT, "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)
from model import DiabetesRiskPredictor   # ml/model.py — DO NOT modify

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
log = logging.getLogger("MediShield.Strategy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)


# ---------------------------------------------------------------------------
# Helper — weighted FedAvg aggregation
# ---------------------------------------------------------------------------

def _fedavg_aggregate(
    results: List[Tuple[List[np.ndarray], int]]
) -> List[np.ndarray]:
    """
    Simple Federated Averaging — equal contribution per hospital.

        global_weights = (w_A + w_B + w_C) / N_hospitals

    Each hospital contributes 1/N regardless of how many patients it has.
    This treats every hospital as an equal participant, which is the
    fairness model preferred for healthcare federation where data volume
    should not dominate clinical influence.

    (The classical sample-weighted FedAvg variant
       w_global = Σ (n_i / Σn_j) · w_i
     is kept as a comment below for reference.)

    Parameters
    ----------
    results : list of (weights, num_examples) tuples — num_examples kept
              for telemetry, not used in this aggregation rule.

    Returns
    -------
    aggregated_weights : list of numpy arrays with same shapes as input.
    """
    n_hospitals = len(results)
    if n_hospitals == 0:
        raise ValueError("Cannot aggregate empty results list.")

    # Allocate zero tensors matching the architecture of the first client
    aggregated = [np.zeros_like(w) for w in results[0][0]]

    # Simple sum across hospitals, then divide by N
    for weights, _num_examples in results:
        for i, w in enumerate(weights):
            aggregated[i] += w

    for i in range(len(aggregated)):
        aggregated[i] /= n_hospitals

    # --- Reference: classical sample-weighted FedAvg (NOT used here) -----
    # total = sum(n for _, n in results)
    # for w, n in results:
    #     contribution = n / total
    #     ...
    # ---------------------------------------------------------------------

    return aggregated


# ---------------------------------------------------------------------------
# MediShieldFedAvg
# ---------------------------------------------------------------------------

class MediShieldFedAvg(fl.server.strategy.Strategy):
    """
    FedAvg strategy with per-round status logging and JSON output.

    The JSON output (federated/status/federated_status.json) is written
    after every round and can be polled by the frontend dashboard.

    Parameters
    ----------
    num_rounds          : total number of FL rounds (for status reporting)
    min_fit_clients     : minimum hospitals needed to start a training round
    min_evaluate_clients : minimum hospitals needed to run evaluation
    min_available_clients : minimum hospitals that must be online
    local_epochs        : number of local training epochs per round
    """

    def __init__(
        self,
        num_rounds: int = 10,
        min_fit_clients: int = 3,
        min_evaluate_clients: int = 3,
        min_available_clients: int = 3,
        local_epochs: int = 5,
    ) -> None:
        super().__init__()

        self.num_rounds = num_rounds
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.local_epochs = local_epochs

        # Track progression over rounds
        self.round_history: List[Dict] = []
        self.current_global_accuracy: float = 0.0
        self.current_round: int = 0

        # Per-hospital metrics collected during the last fit round
        self._last_hospital_metrics: Dict[str, dict] = {}

        # Initial global model weights (set via initialize_parameters)
        self._initial_weights: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------
    # initialize_parameters — called once before round 1
    # ------------------------------------------------------------------
    def initialize_parameters(
        self, client_manager: fl.server.ClientManager
    ) -> Optional[Parameters]:
        """
        Return initial global model parameters.

        We return None, which tells Flower to request parameters from
        one of the clients (it will call get_parameters on client 0).
        This initialises the global model with random weights from one
        hospital's freshly-created DiabetesRiskPredictor.
        """
        log.info("Initialising global model (requesting from first available client).")
        return None     # Flower will call get_parameters on a client

    # ------------------------------------------------------------------
    # configure_fit — tell clients how to train this round
    # ------------------------------------------------------------------
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Select all available clients and send them the current global model."""
        self.current_round = server_round

        # Wait until all expected clients are available
        # (each client holds a DiabetesRiskPredictor trained on its private data)
        client_manager.wait_for(self.min_available_clients)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients,
        )

        # Config sent to every client's fit() method
        config = {"local_epochs": self.local_epochs}
        fit_ins = FitIns(parameters, config)

        log.info(
            f"\n{'='*60}\n"
            f"  ROUND {server_round} / {self.num_rounds} — TRAINING PHASE\n"
            f"  Participating hospitals: "
            + ", ".join(HOSPITAL_DISPLAY.get(f"hospital_{chr(ord('a') + i)}", f"Client {i}")
                        for i in range(len(clients)))
            + f"\n{'='*60}"
        )

        return [(client, fit_ins) for client in clients]

    # ------------------------------------------------------------------
    # aggregate_fit — FedAvg weight aggregation
    # ------------------------------------------------------------------
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate model weights from all clients using simple FedAvg.

        The server receives weight arrays from each hospital, computes
        their simple average ((w_A + w_B + w_C) / 3), and that becomes
        the new global model.  No raw patient data is involved — only
        floating-point weight tensors.
        """
        if not results:
            log.warning(f"Round {server_round}: no results to aggregate.")
            return None, {}

        if failures:
            log.warning(f"Round {server_round}: {len(failures)} client(s) failed.")

        # ------------------------------------------------------------------
        # Collect weights and sample counts from each client result
        # ------------------------------------------------------------------
        weights_results = []
        participating_hospitals = []
        for _, fit_res in results:
            weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            metrics = fit_res.metrics or {}
            weights_results.append((weights, num_examples))

            # Store per-hospital metrics for the status JSON
            hospital_name = metrics.get("hospital_name", "unknown")
            self._last_hospital_metrics[hospital_name] = {
                "local_accuracy": metrics.get("local_accuracy", 0.0),
                "patients_count": metrics.get("patients_count", num_examples),
                "display_name":   metrics.get("display_name", hospital_name),
                "full_name":      metrics.get("full_name", hospital_name),
                "specialty":      metrics.get("specialty", ""),
            }
            participating_hospitals.append(
                metrics.get("display_name", hospital_name)
            )

        # ------------------------------------------------------------------
        # FedAvg — simple average across all hospitals
        # ------------------------------------------------------------------
        aggregated_weights = _fedavg_aggregate(weights_results)

        log.info(
            f"Round {server_round}: aggregated weights from "
            f"{len(results)} hospital(s) → "
            f"{', '.join(participating_hospitals)}"
        )
        log.info(
            f"   FedAvg formula : global = ({' + '.join(participating_hospitals)}) "
            f"/ {len(results)}"
        )
        log.info(
            "   ✅ Only model weights aggregated — no patient data was transmitted."
        )

        # ------------------------------------------------------------------
        # Persist the aggregated global model to federated/saved_models/
        # ------------------------------------------------------------------
        try:
            reference_model = DiabetesRiskPredictor(input_size=8)
            global_path = save_global_model_weights(aggregated_weights, reference_model)
            log.info(f"   💾 Global model checkpoint saved → {global_path}")
        except Exception as exc:
            log.warning(f"   Could not save global model checkpoint: {exc}")

        # ------------------------------------------------------------------
        # Write an interim status (accuracy will be updated in aggregate_evaluate)
        # ------------------------------------------------------------------
        self._write_status(
            server_round=server_round,
            aggregation_status="aggregating",
            training_status="active",
        )

        return ndarrays_to_parameters(aggregated_weights), {}

    # ------------------------------------------------------------------
    # configure_evaluate — tell clients to evaluate the new global model
    # ------------------------------------------------------------------
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.ClientManager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Send the aggregated global model to all clients for evaluation."""
        clients = client_manager.sample(
            num_clients=self.min_evaluate_clients,
            min_num_clients=self.min_evaluate_clients,
        )
        eval_ins = EvaluateIns(parameters, {})
        return [(client, eval_ins) for client in clients]

    # ------------------------------------------------------------------
    # aggregate_evaluate — compute global accuracy from client reports
    # ------------------------------------------------------------------
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Compute weighted global accuracy from all hospital evaluations.

        Each hospital evaluates the global model on its own test set and
        returns (loss, num_test_examples, {accuracy}). We compute the
        weighted average accuracy as the global metric.
        """
        if not results:
            return None, {}

        # Weighted average loss and accuracy
        total_examples = sum(r.num_examples for _, r in results)
        weighted_loss = sum(
            r.loss * r.num_examples for _, r in results
        ) / total_examples

        weighted_acc = sum(
            (r.metrics.get("accuracy", 0.0) * r.num_examples)
            for _, r in results
        ) / total_examples

        self.current_global_accuracy = weighted_acc

        # Record this round in history
        self.round_history.append({
            "round": server_round,
            "accuracy": round(weighted_acc * 100, 2),
        })

        log.info(
            f"Round {server_round} — Global accuracy: {weighted_acc * 100:.2f}%  "
            f"(weighted avg loss: {weighted_loss:.4f})"
        )

        # ------------------------------------------------------------------
        # Write final round status (accuracy now available)
        # ------------------------------------------------------------------
        is_last_round = (server_round == self.num_rounds)
        self._write_status(
            server_round=server_round,
            aggregation_status="completed" if is_last_round else "in_progress",
            training_status="completed" if is_last_round else "active",
        )

        if is_last_round:
            log.info(
                f"\n{'='*60}\n"
                f"  FEDERATED LEARNING COMPLETE\n"
                f"  Final global accuracy : {weighted_acc * 100:.2f}%\n"
                f"  Total rounds          : {server_round}\n"
                f"{'='*60}"
            )

        return weighted_loss, {"global_accuracy": weighted_acc}

    # ------------------------------------------------------------------
    # evaluate — optional server-side evaluation (we use client-side only)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation is skipped; we rely on client evaluation."""
        return None

    # ------------------------------------------------------------------
    # Internal helper — build and write the status JSON
    # ------------------------------------------------------------------
    def _write_status(
        self,
        server_round: int,
        aggregation_status: str,
        training_status: str,
    ) -> None:
        """Build the full status dict and write it to disk."""

        # Build per-hospital section from the most recent fit metrics
        hospital_list = []
        hospital_keys = ["hospital_a", "hospital_b", "hospital_c"]
        for key in hospital_keys:
            metrics = self._last_hospital_metrics.get(key, {})
            info = HOSPITAL_INFO.get(key, {})
            hospital_list.append({
                "name":            info.get("short_name", HOSPITAL_DISPLAY.get(key, key)),
                "display_name":    metrics.get("full_name",  info.get("display_name", key)),
                "specialty":       metrics.get("specialty",  info.get("specialty", "")),
                "local_accuracy":  round(metrics.get("local_accuracy", 0.0) * 100, 2),
                "patients_count":  metrics.get("patients_count", 0),
                "training_status": "active" if training_status == "active" else "idle",
                "sync_status":     "synced" if aggregation_status in ("completed", "in_progress") else "pending",
            })

        status = {
            "current_round":      server_round,
            "total_rounds":       self.num_rounds,
            "global_accuracy":    round(self.current_global_accuracy * 100, 2),
            "training_status":    training_status,
            "aggregation_status": aggregation_status,
            "aggregation_method": "FedAvg (simple average across hospitals)",
            "privacy_guarantee":  "Patient data never leaves hospitals — only model weights are shared.",
            "connected_hospitals": hospital_list,
            "round_history":       self.round_history,
        }

        try:
            save_status(status)
            log.info(
                f"Status written — round {server_round}, "
                f"global_accuracy={status['global_accuracy']}%"
            )
        except Exception as exc:
            log.warning(f"Could not write status JSON: {exc}")
