# MLFlow Lab Detailed Readme

## Overview

This lab performs end to end ML experiment tracking using MLflow on a pizza sales dataset. The purpose is to explore the data, engineer features, train a baseline model, log metrics and artifacts to MLflow, and finally register and transition a model to Production in the MLflow Model Registry.

## Folder Structure

- Data/raw contains the raw dataset pizza_sales.csv
- Data/processed contains the processed dataset daily_sales.csv
- Notebooks contains the main Jupyter notebook pizza_sales_mlflow_lab.ipynb
- Notebooks/mlruns contains MLflow tracking folders and artifacts created during experiment logging
- outputs folder contains:
  - eda_plots which includes all EDA graphs generated
  - feature_importance.csv created after model training
  - metrics.json containing experiment metrics

## MLflow Tracking URI

MLflow is configured to track runs locally on disk using:

file:/Users/sriks/Documents/Projects/MLOps_Submissions/Labs/Lab_8/Experiment_Tracking_Labs/MLFlow_Lab/Notebooks/mlruns

All experiments and registered models are stored within this folder.

## Steps Performed

### 1. Data Loading and Inspection

The raw order level pizza dataset is loaded, inspected and cleaned.

### 2. Data Cleaning

Order date and time are converted into correct datetime formats. Basic numeric columns are cast and essential columns are enforced to be valid.

### 3. Feature Engineering

Additional features are created from the order date including year, month, day, weekday, and weekend indicator. Category columns are cast accordingly.

### 4. Exploratory Data Analysis and Plots

Several exploratory plots are generated to understand patterns in sales:

Daily revenue trend
Top pizzas by revenue
Revenue by weekday
Total price distribution by pizza size

Example plot images are available in:

outputs/eda_plots

### 5. Aggregate to Daily Revenue

The dataset is aggregated from transaction level to daily revenue level. Features like orders per day, lines per day, and date derived features are included.

### 6. Train Test Split

Daily features are used to split into training and testing.

### 7. Model Training and MLflow Logging

A RandomForestRegressor is trained. Parameters, metrics, feature importance and artifacts are logged to MLflow.

This generates in outputs:
- feature_importance.csv
- metrics.json

### 8. Model Registry

A registered model named pizza_sales_regressor is created in the MLflow Model Registry. The best run is used to register version 1. The model is transitioned to Production stage.

### 9. Load Production Model

Finally, the production model is loaded back from the registry and predictions are performed.

## Outputs

All generated plots are stored under:

outputs/eda_plots

Files include:
- sales_trend.png
- top_pizzas.png
- weekday_sales.png
- size_vs_revenue.png

## Key Metrics

From metrics.json:

rmse: 124.38
r2: 0.909

These are logged into MLflow and stored in outputs/metrics.json.

## Conclusion

This lab demonstrates end to end experiment tracking including EDA, model training, artifact logging, metric logging and production model registration using MLflow Model Registry locally. The directory structure contains all artifacts needed to reproduce work and inspect results. 