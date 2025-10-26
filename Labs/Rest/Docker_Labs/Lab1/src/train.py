# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    data_path = "data/csgo_players.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Dataset not found. Please put 'csgo_players.csv' in the data/ folder.")

    # Load data
    df = pd.read_csv(data_path)

    # ---- Basic Cleaning ----
    # Keep only the columns we need
    df = df[["total_kills", "total_deaths", "rating"]].copy()

    # Drop missing values
    df = df.dropna(subset=["total_kills", "total_deaths", "rating"])

    # Ensure numeric types
    df["total_kills"] = pd.to_numeric(df["total_kills"], errors="coerce")
    df["total_deaths"] = pd.to_numeric(df["total_deaths"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # Drop rows that couldn’t convert properly
    df = df.dropna(subset=["total_kills", "total_deaths", "rating"])

    # ---- Feature Engineering ----
    df["KD_ratio"] = df["total_kills"] / (df["total_deaths"] + 1)
    df["High_Performer"] = (df["rating"] > 1.0).astype(int)

    # Select features and target
    X = df[["total_kills", "total_deaths", "KD_ratio"]]
    y = df["High_Performer"]

    # ---- Train/Test Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---- Model Training ----
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # ---- Evaluate ----
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")

    # ---- Save Model ----
    joblib.dump(model, "data/csgo_model.pkl")
    print("✅ Model saved as data/csgo_model.pkl")

if __name__ == "__main__":
    main()
