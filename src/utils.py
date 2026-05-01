import sys
from pathlib import Path
import logging
from datetime import datetime


def create_logger(log_file_path: str):
    """
    Creates a configured logger that logs to both file and console.
    Prevents duplicate handlers on reload.
    """
    Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("fraud_detection_logger")
    logger.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            "%Y-%m-%d %H:%M:%S"
        )

        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


def ensure_directory(path: str):
    """Ensure directory exists, otherwise create it."""
    Path(path).mkdir(parents=True, exist_ok=True)


def timestamp():
    """Return readable timestamp string."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def validate_input(features, required_length=30):
    """
    Validate incoming feature vector for API or inference.
    Ensures:
    - Must be list
    - Must be numeric
    - Must have exact number of features
    """

    if not isinstance(features, list):
        return False, "Input must be a list."

    if len(features) != required_length:
        return False, f"Expected {required_length} features, got {len(features)}."

    if not all(isinstance(x, (int, float)) for x in features):
        return False, "All features must be numeric."

    return True, "Valid input."