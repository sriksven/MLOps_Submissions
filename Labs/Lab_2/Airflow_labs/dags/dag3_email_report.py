from datetime import datetime
from pathlib import Path
import os
import sys

from airflow import DAG
# ✅ Modern import for Airflow ≥ 2.7
from airflow.providers.standard.operators.python import PythonOperator

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

from src.utils.email_utils import send_email_gmail  # noqa: E402

# ---- Email config ----
TO_EMAIL = os.environ.get("ALAB_TO_EMAIL", "sriks071@gmail.com")
FROM_EMAIL = os.environ.get("ALAB_FROM_EMAIL", "sriks071@gmail.com")
SMTP_PASS = os.environ.get("ALAB_SMTP_PASS", "tltfbdggmzkifzyb")  # Gmail App Password

default_args = {"owner": "airflow"}

with DAG(
    dag_id="dag3_email_report",
    description="Send the model training report via email with inline images",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule=None,  # triggered by dag2
    catchup=False,
    tags=["lab2", "email"],
) as dag:

    def _email_report(**context):
        # Prefer the value passed from DAG 2; keep a backwards-compatible fallback.
        dag_run = context.get("dag_run")
        conf = dag_run.conf if dag_run else {}  # may be None when run manually
        report_dir_str = (
            (conf or {}).get("report_dir")  # ✅ matches dag2 now
            or (conf or {}).get("report_dir_xcom_from_dag2")  # legacy fallback
        )

        if report_dir_str:
            report_dir = Path(report_dir_str)
        else:
            # Fallback: choose the newest folder under src/reports
            reports_root = PROJECT_ROOT / "src" / "reports"
            candidates = sorted(
                [p for p in reports_root.iterdir() if p.is_dir()],
                reverse=True
            )
            if not candidates:
                raise RuntimeError("No report directories found under src/reports.")
            report_dir = candidates[0]

        report_html = report_dir / "report.html"
        coefficients_png = report_dir / "coefficients.png"
        actual_png = report_dir / "actual_vs_pred.png"
        residuals_png = report_dir / "residuals.png"
        metrics_json = report_dir / "metrics.json"
        data_head = report_dir / "data_head.csv"

        html_body = report_html.read_text()
        subject = f"[Airflow Lab 2] Model Report – {report_dir.name}"

        send_email_gmail(
            subject=subject,
            html_body=html_body,
            to_email=TO_EMAIL,
            from_email=FROM_EMAIL,
            app_password=SMTP_PASS,
            attachments=[metrics_json, data_head],
            inline_images=[coefficients_png, actual_png, residuals_png],
        )

        print(f"✅ Email sent with report from: {report_dir}")

    email_task = PythonOperator(
        task_id="email_report",
        python_callable=_email_report,
    )
