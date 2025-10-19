## IMDB Sentiment Analysis Pipeline with GitHub Actions

### Overview

This project implements an end-to-end machine learning pipeline for sentiment analysis using the IMDB movie reviews dataset. The workflow is fully automated with GitHub Actions, meaning every code push or scheduled run automatically triggers model training, evaluation, and testing.
The trained models, metrics, and results are versioned and committed back to the repository for tracking and reproducibility.

---

### Key Objectives

- Automate model training and versioning using GitHub Actions.

- Train a Logistic Regression model on the IMDB dataset from Hugging Face.

- Ensure balanced sampling for both sentiment classes.

- Store performance metrics and artifacts for every run.

- Reuse the same TF-IDF vectorizer during testing for consistent results.

- Log model performance with MLflow for experiment tracking.

---

```bash
.github/
 └── workflows/
      └── imdb_train_lr.yml        # GitHub Actions workflow

Labs/
 └── Lab_4/
      └── Github_labs/
           └── Lab_2/
                └── LR/
                     ├── src/
                     │    ├── train_model_lr.py      # Training script
                     │    ├── evaluate_model.py      # Model evaluation
                     │    └── test_model.py          # Model testing
                     └── requirements.txt            # Dependencies

models/                # Saved models and vectorizers
metrics/               # JSON metrics reports
results/               # Test predictions on unseen samples
```


### How the Pipeline Works
1. Workflow Trigger

 - The workflow defined in .github/workflows/imdb_train_lr.yml is triggered on:

 - Every push to the main branch, or

A scheduled daily run (cron job).

2. Environment Setup

GitHub Actions sets up a clean Ubuntu environment with Python 3.9, installs dependencies from requirements.txt, and prepares the workspace.

3. Training Phase

 - The script train_model_lr.py:

 - Loads the IMDB dataset using Hugging Face.

 - Creates a balanced subset with equal numbers of positive and negative samples.

 - Transforms the text data using TF-IDF vectorization (max_features=2000).

 - Splits data into training and testing sets with stratification.

 - Trains a Logistic Regression classifier.

 - Logs Accuracy and F1 Score with MLflow.

 - Saves both the trained model and the TF-IDF vectorizer to the models/ directory using timestamp-based filenames.

Saved Artifacts:
```bash
models/model_<timestamp>_lr.joblib
models/vectorizer_<timestamp>.joblib
```

4. Evaluation Phase

The script evaluate_model.py:

- Loads the latest trained model.

- Runs evaluation using accuracy and F1 score.

- Writes metrics to a JSON file stored in metrics/.

```bash
{
  "accuracy": 0.8791,
  "f1_score": 0.8756
}
```

5. Testing Phase

The script test_model.py:

- Loads both the model and the corresponding TF-IDF vectorizer from models/.

- Loads unseen IMDB test samples from Hugging Face.

- Transforms the new text data using the same vectorizer.

- Predicts sentiment labels.

- Saves prediction results (review excerpt, predicted label, actual label) to results/.

```bash
[
  {
    "sample_id": 0,
    "review_excerpt": "This movie was an excellent drama with great acting...",
    "predicted_label": 1,
    "actual_label": 1
  }
]
```

6. Git Commit Step

- After all steps complete successfully, the workflow:

- Adds new files from models/, metrics/, and results/.

- Commits them with a timestamp message.

- Pushes back to the same GitHub repository automatically.

---

Workflow File Description

Path: .github/workflows/imdb_train_lr.yml

Main Sections:

- Setup: Defines Python version and installs dependencies.

- raining: Runs train_model_lr.py with a generated timestamp.

- Evaluation: Runs evaluate_model.py using the same timestamp.

- Testing: Runs test_model.py to generate predictions.

- Commit and Push: Commits all new outputs to GitHub using the GitHub Actions bot credentials.

---


