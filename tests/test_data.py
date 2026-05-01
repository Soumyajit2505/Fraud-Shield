# tests/test_data.py

from api.predict import predict
import pytest

def test_predict_input_type():
    """
    Non-dictionary input should raise ValueError
    """
    with pytest.raises(ValueError):
        predict("not a dict")
    with pytest.raises(ValueError):
        predict(12345)

def test_predict_missing_amount():
    """
    Missing 'amount' defaults to 0 → Not Fraud
    """
    data = {"type": "transfer"}
    result = predict(data)
    assert result == 0

def test_predict_extra_fields():
    """
    Extra fields should not break predict()
    """
    data = {"amount": 6000, "type": "transfer", "user_id": 123, "location": "NY"}
    result = predict(data)
    assert result == 1