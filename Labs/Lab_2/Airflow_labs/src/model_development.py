from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")  # ✅ prevents macOS GUI crash inside Airflow workers
import matplotlib.pyplot as plt
import joblib
import os

from .utils.paths import DATA_FILE, REPORTS_DIR, MODEL_FILE


# ============================================================
#  Synthetic Data Generation
# ============================================================

def generate_synthetic_dataset(rows: int = 1000, seed: int = 42) -> Path:
    """
    Creates a 1000x10 dataset: 9 numeric features + 1 categorical ("channel") + target "sales".
    Saves to DATA_FILE and returns the path.
    """
    rng = np.random.default_rng(seed)

    # numeric features
    X = rng.normal(0, 1, size=(rows, 9))
    coefs = np.array([2.5, -1.2, 0.7, 0.0, 3.2, -2.2, 1.0, 0.5, -0.8])
    noise = rng.normal(0, 1.0, size=(rows,))
    y = X @ coefs + 10 + noise

    cols = [f"X{i}" for i in range(1, 10)]
    df = pd.DataFrame(X, columns=cols)
    df["channel"] = rng.choice(["search", "social", "display"], size=rows)
    df["sales"] = y

    # ensure total columns = 10 (8 numeric + 1 categorical + 1 target)
    if len(df.columns) > 10:
        df.drop(columns=["X9"], inplace=True)

    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"✅ Dataset generated at: {DATA_FILE}")
    return DATA_FILE


# ============================================================
#  Training / Evaluation
# ============================================================

@dataclass
class TrainResult:
    metrics: Dict[str, float]
    coef: np.ndarray
    feature_names: List[str]
    report_dir: Path


def _prep_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Split features/target and one-hot encode 'channel'."""
    y = df["sales"]
    X = df.drop(columns=["sales"])
    if "channel" in X.columns:
        X = pd.get_dummies(X, columns=["channel"], drop_first=True)
    return X, y, list(X.columns)


def train_and_evaluate(random_state: int = 42, test_size: float = 0.2) -> TrainResult:
    """Train Linear Regression, generate metrics + visualizations."""
    df = pd.read_csv(DATA_FILE)
    X, y, feature_names = _prep_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    # save model
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    # create unique report directory
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report_dir = REPORTS_DIR / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Visualizations ----------

    # (1) Coefficients
    coef = model.coef_
    plt.figure(figsize=(6, 4))
    order = np.argsort(np.abs(coef))[::-1]
    plt.bar([feature_names[i] for i in order], coef[order])
    plt.xticks(rotation=45, ha="right")
    plt.title("Linear Regression Coefficients")
    plt.tight_layout()
    coef_png = report_dir / "coefficients.png"
    plt.savefig(coef_png, dpi=150)
    plt.close()

    # (2) Actual vs Predicted
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, y_pred, s=10, alpha=0.7)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout()
    avp_png = report_dir / "actual_vs_pred.png"
    plt.savefig(avp_png, dpi=150)
    plt.close()

    # (3) Residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30, edgecolor="black")
    plt.title("Residuals Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.tight_layout()
    resid_png = report_dir / "residuals.png"
    plt.savefig(resid_png, dpi=150)
    plt.close()

    # ---------- Metrics & HTML ----------

    metrics = {"r2": float(r2), "rmse": rmse, "mse": float(mse)}
    metrics_json = report_dir / "metrics.json"
    with open(metrics_json, "w") as f:
        json.dump(metrics, f, indent=2)

    html = f"""
    <html>
    <body>
      <h2>Model Training Report</h2>
      <p><b>Run ID:</b> {run_id}</p>
      <h3>Metrics</h3>
      <ul>
        <li>R²: {r2:.4f}</li>
        <li>RMSE: {rmse:.4f}</li>
      </ul>
      <h3>Visualizations</h3>
      <p><img src="{{{{cid:coefficients.png}}}}" width="600"/></p>
      <p><img src="{{{{cid:actual_vs_pred.png}}}}" width="600"/></p>
      <p><img src="{{{{cid:residuals.png}}}}" width="600"/></p>
    </body>
    </html>
    """
    report_html = report_dir / "report.html"
    report_html.write_text(html)

    # Also export dataset head for email attachment
    head_csv = report_dir / "data_head.csv"
    df.head(20).to_csv(head_csv, index=False)

    print(f"✅ Training complete. Report saved under {report_dir}")
    return TrainResult(metrics=metrics, coef=coef, feature_names=feature_names, report_dir=report_dir)
