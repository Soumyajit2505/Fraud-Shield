import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    roc_auc_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ModelTrainer:

    def __init__(self):
        self.data_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"
        self.model_path = PROJECT_ROOT / "models" / "fraud_model.pkl"
        self.threshold_path = PROJECT_ROOT / "models" / "threshold.pkl"
        self.feature_names_path = PROJECT_ROOT / "models" / "feature_names.pkl"

    def load_processed_data(self):
        if not self.data_path.exists():
            raise FileNotFoundError(f"[ERROR] Processed data not found at: {self.data_path}")

        print(f"[INFO] Loading processed dataset...")
        df = pd.read_csv(self.data_path)

        X = df.drop("Class", axis=1)
        y = df["Class"]
        return X, y

    def train_lightgbm(self, X_train, y_train):
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": -1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": 42
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        return model

    def find_best_threshold(self, y_true, y_scores):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores)]

        return best_threshold

    def train(self):
        X, y = self.load_processed_data()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        print("[INFO] Training LightGBM model...")
        model = self.train_lightgbm(X_train, y_train)

        y_scores = model.predict_proba(X_test)[:, 1]
        best_threshold = self.find_best_threshold(y_test, y_scores)

        auc = roc_auc_score(y_test, y_scores)
        print(f"[INFO] ROC-AUC: {auc:.4f}")
        print(f"[INFO] Best Threshold: {best_threshold:.4f}")

        # Save model + threshold + feature names
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, self.model_path)
        joblib.dump(best_threshold, self.threshold_path)
        joblib.dump(list(X.columns), self.feature_names_path)

        print("[INFO] Saved model & threshold successfully")

        print("\n[CLASSIFICATION REPORT]")
        print(classification_report(y_test, (y_scores >= best_threshold).astype(int)))

        return model, best_threshold


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()