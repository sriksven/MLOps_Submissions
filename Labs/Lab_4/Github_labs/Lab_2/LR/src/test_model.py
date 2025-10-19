import os
import json
import argparse
from datasets import load_dataset
from joblib import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    model_path = f"models/model_{timestamp}_lr.joblib"
    vectorizer_path = f"models/vectorizer_{timestamp}.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"❌ Vectorizer not found at {vectorizer_path}")

    print(f"✅ Loaded model from {model_path}")
    print(f"✅ Loaded vectorizer from {vectorizer_path}")

    model = load(model_path)
    vectorizer = load(vectorizer_path)

    print("📥 Loading IMDB test dataset...")
    dataset = load_dataset("imdb")
    test_data = dataset["test"]

    # Use a few unseen reviews for demonstration
    X_samples = test_data["text"][:5]
    y_true = test_data["label"][:5]

    print("🧪 Making predictions...")
    X_vec = vectorizer.transform(X_samples)
    predictions = model.predict(X_vec)

    results = []
    for i, (text, pred, true) in enumerate(zip(X_samples, predictions, y_true)):
        results.append({
            "sample_id": i,
            "review_excerpt": text[:100] + "...",
            "predicted_label": int(pred),
            "actual_label": int(true)
        })

    os.makedirs("results", exist_ok=True)
    result_path = f"results/{timestamp}_test_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Test predictions saved to {result_path}")
    print(json.dumps(results, indent=4))
