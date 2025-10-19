import os
import argparse
import datetime
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from joblib import dump
import mlflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    print("📥 Loading IMDB dataset...")

    # ✅ Load only labeled data (avoid unsupervised subset)
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]

    # Use small portions for fast GitHub Actions run
    X = train_data["text"][:2000] + test_data["text"][:500]
    y = train_data["label"][:2000] + test_data["label"][:500]

    # ✅ Ensure both classes exist
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print("⚠️ Only one class detected, expanding sample...")
        X = train_data["text"][:4000] + test_data["text"][:1000]
        y = train_data["label"][:4000] + test_data["label"][:1000]

    print(f"✅ Classes found: {np.unique(y)}")

    print("🔠 Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_vec = vectorizer.fit_transform(X)

    print("✂️ Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    print("⚙️ Training Logistic Regression model...")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    print("📈 Evaluating model performance...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # ✅ Log metrics with MLflow
    mlflow.set_tracking_uri("./mlruns")
    experiment_name = f"IMDB_LogReg_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    exp_id = mlflow.create_experiment(experiment_name)
    with mlflow.start_run(experiment_id=exp_id, run_name="IMDB_LogisticRegression"):
        mlflow.log_params({
            "model": "LogisticRegression",
            "dataset": "IMDB",
            "vectorizer": "TF-IDF",
            "features": 2000
        })
        mlflow.log_metrics({
            "accuracy": acc,
            "f1_score": f1
        })

    # ✅ Save model with timestamp
    if not os.path.exists("models/"):
        os.makedirs("models/")

    model_filename = f"models/model_{timestamp}_lr.joblib"
    dump(model, model_filename)
    print(f"💾 Model saved as {model_filename}")

    print("🎉 Training completed successfully!")
