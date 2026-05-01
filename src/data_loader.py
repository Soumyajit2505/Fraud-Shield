import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class DataLoader:
    def __init__(self):
        self.raw_path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

    def load_raw_data(self):
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at: {self.raw_path}")

        print(f"[INFO] Loading raw data from → {self.raw_path}")
        return pd.read_csv(self.raw_path)