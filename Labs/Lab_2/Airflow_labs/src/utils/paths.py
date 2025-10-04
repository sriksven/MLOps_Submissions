import os
from pathlib import Path

# Default to the absolute path you gave, but allow override with env var
PROJECT_ROOT = Path(os.environ.get(
    "AIRFLOW_LAB2_ROOT",
    "/Users/sriks/Documents/Projects/MLOps_Submissions/Labs/Lab_2/Airflow_labs"
)).resolve()

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
MODELS_DIR = SRC_DIR / "models"
REPORTS_DIR = SRC_DIR / "reports"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "sales_1000x10.csv"
MODEL_FILE = MODELS_DIR / "linear_regression.pkl"
