"""
server.py
=========
Federated learning orchestrator — the entry point for the entire simulation.

Run this file to start a full federated training session:
    python federated/server.py

What happens
------------
1. split_data.split_and_save()  — create hospital CSV partitions if needed
2. Write an initial status JSON so the dashboard shows 'starting'
3. Start Flower's local simulation engine with 3 hospital clients
4. The simulation runs num_rounds aggregation rounds:
     a. Server sends global weights to all hospitals
     b. Each hospital trains locally (no data leaves)
     c. Each hospital returns updated weights
     d. Server aggregates via FedAvg → new global model
     e. Server evaluates global model on each hospital's test set
     f. Status JSON is updated after every round
5. Simulation completes; final status written with 'completed' state

Simulation vs. real deployment
--------------------------------
This uses `fl.simulation.start_simulation()` which runs every client
and the server in the same Python process. No real network sockets are
opened. This is intentional for hackathon stability and local demo use.
To move to real deployment, replace this file's simulation call with
`fl.server.start_server()` and run clients on separate machines.
"""

import logging
import os
import sys

import flwr as fl

# ---------------------------------------------------------------------------
# Add project root to sys.path so sibling imports work regardless of cwd
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FEDERATED_DIR = os.path.join(_PROJECT_ROOT, "federated")

for _dir in (_PROJECT_ROOT, _FEDERATED_DIR):
    if _dir not in sys.path:
        sys.path.insert(0, _dir)

# ---------------------------------------------------------------------------
# Local imports (after sys.path is fixed)
# ---------------------------------------------------------------------------
import split_data                       # federated/split_data.py
from client   import make_client         # federated/client.py
from strategy import MediShieldFedAvg    # federated/strategy.py
from utils    import (                   # federated/utils.py
    save_status,
    make_initial_status,
    export_global_to_backend,
    HOSPITAL_DISPLAY,
    HOSPITAL_INFO,
    SAVED_MODELS_DIR,
    GLOBAL_MODEL_PATH,
    BACKEND_MODEL_PATH,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("MediShield.Server")

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
DEFAULT_ROUNDS       = 10    # number of federated aggregation rounds
DEFAULT_LOCAL_EPOCHS = 5     # local training epochs per client per round
NUM_HOSPITALS        = 3     # Hospital A, B, C

HOSPITAL_NAMES = ["hospital_a", "hospital_b", "hospital_c"]


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def run_federated_learning(
    num_rounds: int = DEFAULT_ROUNDS,
    local_epochs: int = DEFAULT_LOCAL_EPOCHS,
) -> None:
    """
    Run the full federated learning simulation.

    Parameters
    ----------
    num_rounds    : number of FL aggregation rounds
    local_epochs  : local training epochs each hospital runs per round
    """

    # ------------------------------------------------------------------
    # Privacy + setup banner
    # ------------------------------------------------------------------
    log.info("")
    log.info("╔" + "═" * 68 + "╗")
    log.info("║" + " MEDISHIELD AI — MULTI-HOSPITAL FEDERATED LEARNING".center(68) + "║")
    log.info("╠" + "═" * 68 + "╣")
    log.info("║" + " PRIVACY GUARANTEE".center(68) + "║")
    log.info("║" + " Patient data NEVER leaves the hospitals.".center(68) + "║")
    log.info("║" + " Only model weights are aggregated centrally.".center(68) + "║")
    log.info("╠" + "═" * 68 + "╣")
    log.info("║" + f" Aggregation rounds : {num_rounds}".ljust(68) + "║")
    log.info("║" + f" Local epochs/round : {local_epochs}".ljust(68) + "║")
    log.info("║" + f" FedAvg formula     : global = (w_A + w_B + w_C) / 3".ljust(68) + "║")
    log.info("╠" + "═" * 68 + "╣")
    log.info("║" + " PARTICIPATING HOSPITALS".center(68) + "║")
    for key in HOSPITAL_NAMES:
        info = HOSPITAL_INFO.get(key, {})
        line = f"  • {info.get('display_name', key)} ({info.get('specialty', '')})"
        log.info("║" + line.ljust(68) + "║")
    log.info("╚" + "═" * 68 + "╝")
    log.info("")

    # ------------------------------------------------------------------
    # STEP 1 — Ensure hospital data partitions exist
    # ------------------------------------------------------------------
    log.info("\n[Step 1] Splitting dataset into hospital partitions...")
    split_data.split_and_save()

    # ------------------------------------------------------------------
    # STEP 2 — Write initial 'starting' status for the dashboard
    # ------------------------------------------------------------------
    log.info("\n[Step 2] Writing initial status JSON...")
    initial_status = make_initial_status(
        total_rounds=num_rounds,
        hospital_names=HOSPITAL_NAMES,
    )
    save_status(initial_status)
    log.info("  Status written. Dashboard can now read federated_status.json.")

    # ------------------------------------------------------------------
    # STEP 3 — Create the FedAvg strategy
    # ------------------------------------------------------------------
    log.info("\n[Step 3] Initialising MediShieldFedAvg strategy...")
    strategy = MediShieldFedAvg(
        num_rounds=num_rounds,
        min_fit_clients=NUM_HOSPITALS,
        min_evaluate_clients=NUM_HOSPITALS,
        min_available_clients=NUM_HOSPITALS,
        local_epochs=local_epochs,
    )

    # ------------------------------------------------------------------
    # STEP 4 — Run the simulation
    # Flower's start_simulation() orchestrates all clients and the server
    # inside this single Python process.
    # ------------------------------------------------------------------
    log.info("\n[Step 4] Starting Flower simulation...\n")

    history = fl.simulation.start_simulation(
        client_fn=make_client,           # factory: Context → fl.client.Client
        num_clients=NUM_HOSPITALS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        # Ray resources per simulated client (CPU-only for hackathon)
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    # ------------------------------------------------------------------
    # STEP 5 — Write final completed status
    # ------------------------------------------------------------------
    log.info("\n[Step 5] Writing final completed status...")

    # The strategy already wrote the last round's status, but we stamp
    # the training_status as 'completed' for clarity.
    from utils import load_status
    final_status = load_status()
    final_status["training_status"]    = "completed"
    final_status["aggregation_status"] = "completed"
    save_status(final_status)

    # ------------------------------------------------------------------
    # STEP 5b — Export final aggregated global model to backend folder
    # so the FastAPI /predict endpoint can serve real predictions from
    # the federated-trained weights.
    # ------------------------------------------------------------------
    log.info("\n[Step 5b] Exporting global model to backend...")
    try:
        dest = export_global_to_backend()
        log.info(f"  ✅ Global model copied → {dest}")
        log.info("  Backend /predict will use these federated-trained weights.")
    except FileNotFoundError as exc:
        log.warning(f"  ⚠  Could not export global model: {exc}")

    # ------------------------------------------------------------------
    # STEP 6 — Print summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("  SIMULATION COMPLETE")
    log.info("=" * 60)

    if history.metrics_distributed:
        # Extract final global accuracy from the last evaluated round
        acc_history = history.metrics_distributed.get("global_accuracy", [])
        if acc_history:
            _, final_acc = acc_history[-1]   # (round_number, accuracy)
            log.info(f"  Final global accuracy : {final_acc * 100:.2f}%")

    log.info(f"  Rounds completed       : {num_rounds}")
    log.info(f"  Hospitals participated : {NUM_HOSPITALS}")
    log.info(
        f"  Status file            : "
        f"{os.path.join(_PROJECT_ROOT, 'federated', 'status', 'federated_status.json')}"
    )
    log.info("")
    log.info("  Saved model artefacts:")
    log.info(f"    • {os.path.join(SAVED_MODELS_DIR, 'hospital_a_local.pth')}")
    log.info(f"    • {os.path.join(SAVED_MODELS_DIR, 'hospital_b_local.pth')}")
    log.info(f"    • {os.path.join(SAVED_MODELS_DIR, 'hospital_c_local.pth')}")
    log.info(f"    • {GLOBAL_MODEL_PATH}  (aggregated)")
    log.info(f"    • {BACKEND_MODEL_PATH}  (served by FastAPI /predict)")
    log.info("")
    log.info("  🔒 Throughout all rounds, raw patient data NEVER left any hospital.")
    log.info("=" * 60)

    _print_accuracy_progression(strategy.round_history)


# ---------------------------------------------------------------------------
# Helper — pretty-print the round-by-round accuracy table
# ---------------------------------------------------------------------------

def _print_accuracy_progression(round_history: list) -> None:
    """Print a formatted table of accuracy improvement across rounds."""
    if not round_history:
        return

    print("\n" + "=" * 40)
    print("  Round-by-round accuracy progression")
    print("=" * 40)
    print(f"  {'Round':>6}  {'Global Accuracy':>16}")
    print("  " + "-" * 26)
    for entry in round_history:
        print(f"  {entry['round']:>6}  {entry['accuracy']:>14.2f}%")
    print("=" * 40 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Allow overriding rounds from the command line:
    #   python server.py 5        → 5 rounds
    #   python server.py 10 3     → 10 rounds, 3 local epochs
    import sys as _sys

    _num_rounds = int(_sys.argv[1]) if len(_sys.argv) > 1 else DEFAULT_ROUNDS
    _local_epochs = int(_sys.argv[2]) if len(_sys.argv) > 2 else DEFAULT_LOCAL_EPOCHS

    run_federated_learning(num_rounds=_num_rounds, local_epochs=_local_epochs)
