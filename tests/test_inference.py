# tests/test_inference.py

from api.predict import predict

def test_rule_based_logic_not_fraud():
    """
    Amount below threshold → Not Fraud
    """
    data = {"amount": 1000, "type": "transfer"}
    assert predict(data) == 0

def test_rule_based_logic_fraud():
    """
    Amount above threshold → Fraud
    """
    data = {"amount": 10000, "type": "transfer"}
    assert predict(data) == 1

def test_boundary_amount():
    """
    Test boundary values around 5000
    """
    data = {"amount": 5000, "type": "transfer"}
    assert predict(data) == 0  # boundary: not fraud

    data = {"amount": 5001, "type": "transfer"}
    assert predict(data) == 1  # just above threshold → fraud