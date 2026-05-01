import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Add project root to path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from api.predict import predict

# Define FastAPI app
app = FastAPI(title="Fraud Detection API", version="1.0")

# Define input data model using Pydantic
class Transaction(BaseModel):
    amount: float
    type: str = "transfer"  # default value if type not provided
    # Add more fields if needed, e.g., location, user_id

@app.get("/")
def root():
    """
    Root endpoint to check API health.
    """
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def get_prediction(transaction: Transaction):
    """
    Endpoint to predict if a transaction is fraudulent.

    Accepts JSON input matching the Transaction model.
    Returns 0 (Not Fraud) or 1 (Fraud).
    """
    try:
        # Convert Pydantic model to dict
        data = transaction.dict()
        prediction = predict(data)
        return {"prediction": prediction}
    except Exception as e:
        # Return HTTP 400 for bad input / errors
        raise HTTPException(status_code=400, detail=str(e))