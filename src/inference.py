# src/inference.py

import sys
from pathlib import Path
import numpy as np
import joblib
import traceback


class FraudInference:
    """
    Inference Pipeline for Fraud Detection:
    - Load model, scaler, and threshold
    - Validate input
    - Predict fraud probability
    - Apply threshold to classify
    """

    def __init__(self):
        # -----------------------------
        # Set project root using pathlib
        # -----------------------------
        self.root = Path(__file__).resolve().parent.parent

        # -----------------------------
        # Paths to artifacts
        # -----------------------------
        self.model_path = self.root / "models" / "fraud_model.pkl"
        self.scaler_path = self.root / "models" / "scaler.pkl"
        self.threshold_path = self.root / "models" / "threshold.pkl"

        # -----------------------------
        # Placeholders for objects
        # -----------------------------
        self.model = None
        self.scaler = None
        self.threshold = None

    # ======================================================
    # Load all artifacts
    # ======================================================
    def load_artifacts(self):
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"❌ Model not found at: {self.model_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"❌ Scaler not found at: {self.scaler_path}")
            if not self.threshold_path.exists():
                raise FileNotFoundError(f"❌ Threshold not found at: {self.threshold_path}")

            print("🔄 Loading model...")
            self.model = joblib.load(self.model_path)

            print("🔄 Loading scaler...")
            self.scaler = joblib.load(self.scaler_path)

            print("🔄 Loading threshold...")
            self.threshold = joblib.load(self.threshold_path)

            print("✅ Model, Scaler & Threshold Loaded Successfully")

        except Exception as e:
            print("❌ Error while loading artifacts:", e)
            traceback.print_exc()

    # ======================================================
    # Validate input
    # ======================================================
    def validate_input(self, data):
        if not isinstance(data, (list, np.ndarray)):
            raise ValueError("Input must be a list or numpy array.")
        if len(data) != 30:
            raise ValueError(f"Expected 30 features, got {len(data)}.")
        return np.array(data).reshape(1, -1)

    # ======================================================
    # Single prediction
    # ======================================================
    def predict_single(self, input_data):
        try:
            # Load artifacts if not loaded
            if self.model is None or self.scaler is None or self.threshold is None:
                self.load_artifacts()

            # Validate and reshape input
            input_data = self.validate_input(input_data)

            # Scale
            input_scaled = self.scaler.transform(input_data)

            # Predict probability
            prob = float(self.model.predict_proba(input_scaled)[0][1])

            # Apply threshold
            prediction = 1 if prob >= self.threshold else 0

            return {
                "fraud_probability": prob,
                "prediction": "Fraud" if prediction == 1 else "Not Fraud",
                "threshold_used": float(self.threshold)
            }

        except Exception as e:
            print("❌ Error during inference:", e)
            traceback.print_exc()
            return {"error": str(e)}


# ======================================================
# Demo run
# ======================================================
if __name__ == "__main__":
    print("\n🚀 Running Inference Demo...\n")

    # Example input (30 features)
    sample_input = [0.1] * 30

    infer = FraudInference()
    output = infer.predict_single(sample_input)

    print("\n=== PREDICTION OUTPUT ===")
    print(output)