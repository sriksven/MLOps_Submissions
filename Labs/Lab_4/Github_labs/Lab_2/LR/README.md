# Automated IMDB Sentiment Analysis Pipeline

This project automates the training and evaluation of a **Logistic Regression** model on the **IMDB movie reviews dataset** using **GitHub Actions**.

### Features
- Pulls real text data from Hugging Face Datasets.
- Trains and evaluates a Logistic Regression classifier.
- Logs experiments to MLflow.
- Automatically commits model and metrics to the repo.

### Workflow
1. Triggered on push or daily schedule.
2. Loads IMDB data.
3. Trains model → evaluates → saves model → logs metrics.
4. Commits results back to GitHub.

### Tech Stack
- Python
- scikit-learn
- Hugging Face Datasets
- MLflow
- GitHub Actions
