from datetime import datetime
from pathlib import Path
import os
import sys

from airflow import DAG
# ✅ Modern imports for Airflow ≥ 2.7
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator

# -------------------------------------------------------------------
# Ensure local imports (src folder)
# -------------------------------------------------------------------
PROJECT_ROOT = Path(
    os.environ.get(
        "AIRFLOW_LAB2_ROOT",
        "/Users/sriks/Documents/Projects/MLOps_Submissions/Labs/Lab_2/Airflow_labs"
    )
).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.model_development import train_and_evaluate  # noqa: E402

# -------------------------------------------------------------------
# DAG definition
# -------------------------------------------------------------------
default_args = {"owner": "airflow"}

with DAG(
    dag_id="dag2_train_model",
    description="Train Linear Regression model and trigger email report DAG",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,          # triggered by dag1
    catchup=False,
    tags=["lab2", "train"],
) as dag:

    def _train(**kwargs):
        """Train model, save artifacts, and return report directory path."""
        result = train_and_evaluate()
        print(f"✅ Model trained successfully. Report saved at: {result.report_dir}")
        return str(result.report_dir)

    train_task = PythonOperator(
        task_id="train_and_report",
        python_callable=_train,
    )

    # ✅ Automatically trigger DAG3 and pass report_dir through XCom
    trigger_email = TriggerDagRunOperator(
        task_id="trigger_dag3_email_report",
        trigger_dag_id="dag3_email_report",
        reset_dag_run=True,
        wait_for_completion=False,
        conf={
            "report_dir": "{{ ti.xcom_pull(task_ids='train_and_report') }}"
        },
    )

    # DAG flow
    train_task >> trigger_email
