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

    print("📥 Loading IMDB dataset (labeled only)...")
    dataset = load_dataset("imdb", download_mode="force_redownload")

    train_data = dataset["train"]
    test_data = dataset["test"]

    # ✅ Manually balance classes (equal positives & negatives)
    pos_train_idx = [i for i, y in enumerate(train_data["label"]) if y == 1][:1000]
    neg_train_idx = [i for i, y in enumerate(train_data["label"]) if y == 0][:1000]
    pos_test_idx = [i for i, y in enumerate(test_data["label"]) if y == 1][:250]
    neg_test_idx = [i for i, y in enumerate(test_data["label"]) if y == 0][:250]

    X = (
        [train_data["text"][i] for i in pos_train_idx + neg_train_idx]
        + [test_data["text"][i] for i in pos_test_idx + neg_test_idx]
    )
    y = (
        [1 for _ in pos_train_idx] + [0 for _ in neg_train_idx]
        + [1 for _ in pos_test_idx] + [0 for _ in neg_test_idx]
    )

    print(f"✅ Class distribution: {np.unique(y, return_counts=True)}")

    print("🔠 Vectorizing text using TF-IDF (2000 features)...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_vec = vectorizer.fit_transform(X)

    print("✂️ Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    print("⚙️ Training Logistic Regression model...")
    model = LogisticRegression(max_iter=300)
    model.fit(X_train, y_train)

    print("📈 Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # ✅ Log metrics with MLflow
    mlflow.set_tracking_uri("./mlruns")
    exp_name = f"IMDB_LogReg_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    exp_id = mlflow.create_experiment(exp_name)
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

    # ✅ Save model and vectorizer
    os.makedirs("models", exist_ok=True)
    model_filename = f"models/model_{timestamp}_lr.joblib"
    vectorizer_filename = f"models/vectorizer_{timestamp}.joblib"
    dump(model, model_filename)
    dump(vectorizer, vectorizer_filename)

    print(f"💾 Model saved as {model_filename}")
    print(f"💾 Vectorizer saved as {vectorizer_filename}")
    print("🎉 Training completed successfully!")
