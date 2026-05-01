import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# ----------------------------------------------------
# Project root using pathlib
# ----------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Now import from src without errors
from src.data_loader import DataLoader


class Preprocessor:
    """
    Handles preprocessing:
    - Load raw data
    - Clean dataset
    - Scale features
    - Train/Test Split
    - SMOTE balancing
    - Save processed data & scaler
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.loader = DataLoader()

    def load_data(self):
        """Load raw credit card fraud dataset"""
        return self.loader.load_raw_data()

    def clean_data(self, df):
        """Remove duplicates + missing values"""
        df = df.drop_duplicates()
        df = df.dropna()
        return df

    def scale_features(self, X):
        """Fit scaler and transform"""
        return self.scaler.fit_transform(X)

    def save_scaler(self):
        """Save fitted scaler"""
        scaler_path = PROJECT_ROOT / "data" / "processed" / "scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"[INFO] Scaler saved → {scaler_path}")

    def preprocess(self):
        """Main preprocessing pipeline"""

        # 1️⃣ Load raw data
        df = self.load_data()

        # 2️⃣ Clean data
        df = self.clean_data(df)

        # 3️⃣ Split features & labels
        X = df.drop("Class", axis=1)
        y = df["Class"]

        # 4️⃣ Scale features
        X_scaled = self.scale_features(X)
        self.save_scaler()

        # 5️⃣ Train-test split before SMOTE
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        # 6️⃣ Apply SMOTE ONLY on training data
        sm = SMOTE(random_state=42)
        X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train)

        # 7️⃣ Save processed full dataset
        processed_path = PROJECT_ROOT / "data" / "processed" / "processed_data.csv"

        processed_df = pd.DataFrame(X_scaled, columns=X.columns)
        processed_df["Class"] = y.values

        processed_path.parent.mkdir(parents=True, exist_ok=True)
        processed_df.to_csv(processed_path, index=False)

        print(f"[INFO] Processed Data Saved → {processed_path}")
        print("[INFO] Preprocessing Completed Successfully")

        return X_train_bal, X_test, y_train_bal, y_test


if __name__ == "__main__":
    pre = Preprocessor()
    pre.preprocess()