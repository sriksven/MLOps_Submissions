# src/predict.py

import joblib
import pandas as pd
import sys
import os

def predict(total_kills, total_deaths):
    model_path = "data/csgo_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Please run train.py first.")

    # Load trained model
    model = joblib.load(model_path)

    # Compute KD ratio from inputs
    kd_ratio = total_kills / (total_deaths + 1)

    # Prepare input features for model
    features = pd.DataFrame(
        [[total_kills, total_deaths, kd_ratio]],
        columns=["total_kills", "total_deaths", "KD_ratio"]
    )

    # Make prediction
    prediction = model.predict(features)[0]
    label = "High Performer" if prediction == 1 else "Average Player"
    print(f"Prediction: {label}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/predict.py <total_kills> <total_deaths>")
        sys.exit(1)

    total_kills = float(sys.argv[1])
    total_deaths = float(sys.argv[2])
    predict(total_kills, total_deaths)
