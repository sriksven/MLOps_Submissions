import os
import json
import argparse
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from joblib import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    print("📥 Loading IMDB test dataset...")
    dataset = load_dataset("imdb", split="test[:500]")
    X = dataset["text"]
    y = dataset["label"]

    print("🔠 Vectorizing text using TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X_vec = vectorizer.fit_transform(X)

    model_path = f"models/model_{timestamp}_lr.joblib"
    model = load(model_path)
    print(f"✅ Loaded model from {model_path}")

    y_pred = model.predict(X_vec)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }

    if not os.path.exists("metrics/"):
        os.makedirs("metrics/")
    with open(f"metrics/{timestamp}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📈 Evaluation complete. Metrics saved to metrics folder.")
