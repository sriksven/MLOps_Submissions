

```bash 
pizza_sales_mlflow/
│
├── data/
│   ├── raw/
│   │   └── pizza_sales.csv                     # Original Kaggle dataset
│   ├── processed/
│   │   └── daily_sales.csv                     # Aggregated daily sales after cleaning
│   └── eda/
│       ├── top_pizzas.png
│       ├── sales_trends.png
│       ├── weekday_sales.png
│       └── size_vs_revenue.png
│
├── notebooks/
│   ├── 01_data_exploration.ipynb               # Exploratory notebook for data understanding
│   ├── 02_feature_engineering_and_eda.ipynb    # EDA + feature creation
│   ├── 03_model_training_mlflow.ipynb          # MLflow-tracked model training
│   └── 04_model_inference_serving.ipynb        # Loading, testing, and serving the model
│
├── scripts/
│   ├── __init__.py
│   ├── load_data.py                            # Load and validate raw data
│   ├── preprocess_data.py                      # Clean, transform, and aggregate daily sales
│   ├── eda_analysis.py                         # All visualization and statistical summaries
│   ├── train_model.py                          # Model training + MLflow logging
│   ├── register_model.py                       # Model registration and promotion to Production
│   ├── inference.py                            # Load and predict using the Production model
│   └── serve_model.sh                          # Command to serve MLflow model locally
│
├── mlruns/                                     # Auto-generated MLflow experiment tracking directory
│
├── models/                                     # Optional: export trained models here
│
├── outputs/
│   ├── metrics.json                            # Saved model evaluation metrics
│   └── feature_importance.csv                  # Ranked feature importances
│
├── logs/
│   └── training_log.txt                        # Logs from each run
│
├── configs/
│   └── params.yaml                             # Model parameters, paths, random seeds, etc.
│
├── requirements.txt                            # All Python dependencies
│
├── README.md                                   # Project overview and instructions
│
├── run_pipeline.py                             # Main orchestrator script (runs all stages sequentially)
│
└── .gitignore
```