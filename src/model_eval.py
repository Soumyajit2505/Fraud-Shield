# src/model_eval.py

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    classification_report
)

# -----------------------------
# Project Root
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# -----------------------------
# Load Model & Threshold
# -----------------------------
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "models" / "threshold.pkl"

try:
    model = joblib.load(MODEL_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    print("[INFO] Model and threshold loaded successfully.")
except FileNotFoundError:
    print(f"[WARNING] Model or threshold file not found at {MODEL_PATH} or {THRESHOLD_PATH}. Using defaults.")
    model = None
    threshold = 0.5  # default threshold

# -----------------------------
# Prediction Function
# -----------------------------
def predict(transaction: dict) -> dict:
    """
    Predict if a transaction is fraud or not.

    Args:
        transaction (dict): Dictionary of transaction features.
            Must match the features used during training.

    Returns:
        dict: {
            "fraud_probability": float,
            "prediction": str ("Fraud"/"Not Fraud"),
            "threshold_used": float
        }
    """
    if model is None:
        # fallback if model is not loaded
        return {"fraud_probability": 0.0, "prediction": "Not Fraud", "threshold_used": threshold}

    # Convert dict to DataFrame (1 row)
    X = pd.DataFrame([transaction])
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Predict probability
    prob = float(model.predict_proba(X)[:, 1][0])
    pred = "Fraud" if prob >= threshold else "Not Fraud"

    return {"fraud_probability": prob, "prediction": pred, "threshold_used": float(threshold)}

# -----------------------------
# Model Evaluation Class
# -----------------------------
class ModelEvaluator:
    """Evaluate model performance on dataset."""

    def __init__(self):
        self.data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
        self.eval_dir = PROJECT_ROOT / "models" / "evaluation"
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"[ERROR] Processed dataset missing at {self.data_path}")
        df = pd.read_csv(self.data_path)
        X = df.drop("Class", axis=1)
        y = df["Class"]
        return X, y

    def evaluate(self):
        X, y = self.load_data()
        if model is None:
            print("[ERROR] Model not loaded. Cannot evaluate.")
            return

        # Predictions
        y_scores = model.predict_proba(X)[:, 1]
        y_pred = (y_scores >= threshold).astype(int)

        # Metrics
        roc_auc = roc_auc_score(y, y_scores)
        print(f"\n[METRIC] ROC-AUC Score: {roc_auc:.4f}")
        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(y, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        disp.figure_.savefig(self.eval_dir / "confusion_matrix.png")
        plt.close(disp.figure_)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y, y_scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid()
        plt.savefig(self.eval_dir / "roc_curve.png")
        plt.close()

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_scores)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.grid()
        plt.savefig(self.eval_dir / "pr_curve.png")
        plt.close()

        print(f"[INFO] Evaluation plots saved in {self.eval_dir}")

        return {"roc_auc": roc_auc, "confusion_matrix": cm}

# -----------------------------
# Run evaluation if executed directly
# -----------------------------
if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate()