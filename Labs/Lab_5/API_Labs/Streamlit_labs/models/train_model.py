import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

def train_salary_model(data_path: str, model_path: str):
    """
    Train a salary prediction model from the preprocessed dataset.

    Steps:
    1. Load preprocessed data
    2. Split into features (X) and target (y)
    3. Train-test split
    4. Train RandomForest model
    5. Evaluate metrics
    6. Save model as .pkl
    """

    # Load the processed data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop(columns=["salary_in_usd"])
    y = df["salary_in_usd"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("✅ Model training complete.")
    print(f"📊 Mean Absolute Error: {mae:,.2f}")
    print(f"📈 R² Score: {r2:.3f}")

    # Create model directory if not exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save model
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")

    return model, mae, r2


if __name__ == "__main__":
    data_path = "data/cleaned_data.csv"
    model_path = "models/salary_model.pkl"
    train_salary_model(data_path, model_path)
