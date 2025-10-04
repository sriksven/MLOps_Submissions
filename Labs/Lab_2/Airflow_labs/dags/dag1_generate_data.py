from datetime import datetime
from pathlib import Path
import os
import sys

from airflow import DAG
# ✅ Modern imports for Airflow ≥ 2.7
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

# -------------------------------------------------------------------
# Ensure local package imports work by adding project root to sys.path
# -------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get(
        "AIRFLOW_LAB2_ROOT",
        "/Users/sriks/Documents/Projects/MLOps_Submissions/Labs/Lab_2/Airflow_labs"
    )
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_development import generate_synthetic_dataset  # noqa: E402

# -------------------------------------------------------------------
# DAG definition
# -------------------------------------------------------------------
default_args = {"owner": "airflow"}

with DAG(
    dag_id="dag1_generate_data",
    description="Generate dataset and trigger model training DAG",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,            # manual trigger only
    catchup=False,
    tags=["lab2", "data", "manual"],
) as dag:

    def _generate_data():
        """Generate synthetic dataset."""
        path = generate_synthetic_dataset(rows=1000, seed=42)
        print(f"✅ Generated dataset at: {path}")

    gen_data = PythonOperator(
        task_id="generate_data",
        python_callable=_generate_data,
    )

    trigger_next = TriggerDagRunOperator(
        task_id="trigger_dag2_train_model",
        trigger_dag_id="dag2_train_model",
        reset_dag_run=True,
        wait_for_completion=False,
    )

    # Task flow
    gen_data >> trigger_next
