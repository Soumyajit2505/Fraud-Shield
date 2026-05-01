# api/predict.py

"""
Fraud Detection Model Inference Module
--------------------------------------
This module contains the logic to predict if a transaction is fraudulent.
It is imported by api/app.py to provide predictions via FastAPI.

Features:
- Safe input handling
- Extensible for future ML models
- Consistent output
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

# Optional: Load real ML model if available
MODEL_PATH = BASE_DIR / "models" / "fraud_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
THRESHOLD_PATH = BASE_DIR / "models" / "threshold.pkl"

model = None
scaler = None
threshold = 0.5

try:
    if MODEL_PATH.exists() and SCALER_PATH.exists() and THRESHOLD_PATH.exists():
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        threshold = joblib.load(THRESHOLD_PATH)
except Exception as e:
    print(f"[WARNING] Could not load model artifacts: {e}. Using fallback logic.")


def predict(transaction: dict) -> int:
    """
    Predict if a transaction is fraudulent.

    Args:
        transaction (dict): A dictionary containing transaction data. Example:
                            {"amount": 100, "type": "transfer"}

    Returns:
        int: 0 = Not Fraud, 1 = Fraud
    """
    try:
        # Input validation
        if not isinstance(transaction, dict):
            raise ValueError("Input must be a dictionary.")

        # Extract fields safely
        amount = transaction.get("amount", 0)
        trans_type = transaction.get("type", "transfer")

        # If model is loaded, use it (requires full feature set)
        if model is not None and scaler is not None:
            # For now, fallback to rule-based since we don't have full features
            # This can be enhanced when full feature set is available
            pass

        # Rule-based placeholder logic
        # Replace with real ML model prediction later
        if amount > 5000:
            return 1  # Fraud
        else:
            return 0  # Not Fraud

    except Exception as e:
        raise ValueError(f"Prediction error: {str(e)}")


# Example usage for testing
if __name__ == "__main__":
    sample_data = {"amount": 10000, "type": "transfer"}
    print("Prediction:", predict(sample_data))