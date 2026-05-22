import os
import sys

# Ensure the ml directory is in the import search path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_DIR = os.path.join(BASE_DIR, "ml")
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)

# Import the complete production app from ml/api.py
# This app contains all routes: /predict, /metrics, /add-patient, /retrain, /health
from ml.api import app

