import joblib
import numpy as np
import csv
from datetime import datetime
from pathlib import Path

# Paths
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"
LOG_PATH = Path(__file__).resolve().parent.parent / "model" / "prediction_log.csv"

def load_model():
    """Load the trained model from disk."""
    return joblib.load(MODEL_PATH)

def predict_data(X):
    """
    Predict the class labels and probabilities for the input data.

    Args:
        X (numpy.ndarray): Input data for which predictions are to be made.
    Returns:
        dict: Contains predicted class and probability scores.
    """
    model = load_model()

    # Predictions
    y_pred = model.predict(X)[0]
    y_proba = model.predict_proba(X)[0]

    # Ensure numpy array -> plain list of floats
    y_proba = [float(p) for p in np.array(y_proba)]

    # Log the request and prediction
    log_prediction(X, y_pred, y_proba)

    return {
        "class": int(y_pred),
        "probabilities": y_proba
    }

def log_prediction(X, y_pred, y_proba):
    """Append input and prediction details into CSV log file."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()

    # Convert everything into serializable Python objects
    X = np.array(X).tolist()

    with open(LOG_PATH, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["timestamp", "features", "predicted_class", "probabilities"])
        writer.writerow([datetime.now().isoformat(), X, int(y_pred), y_proba])
