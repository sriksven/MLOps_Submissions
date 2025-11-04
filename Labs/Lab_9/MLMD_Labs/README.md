# Standalone ML Metadata (MLMD) Lab with Iris Dataset

## Overview

This lab demonstrates the implementation of ML Metadata (MLMD) tracking for a machine learning pipeline using the Iris dataset. The project showcases how to record and retrieve metadata from an end-to-end ML workflow, including data ingestion, model training, and evaluation.

## Objectives

- Implement a standalone MLMD system using SQLite backend
- Track artifacts (datasets, models, metrics) throughout the ML pipeline
- Record executions (data ingestion, training, evaluation) and their properties
- Establish lineage relationships between artifacts and executions
- Query and visualize the complete pipeline metadata

## Environment Setup

### Python Version
- Python 3.10.14

### Required Dependencies
```
tensorflow
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyterlab
ipykernel
```

### Installation
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyterlab ipykernel
```

Note: Due to compatibility issues with `ml-metadata` on ARM-based macOS systems, this implementation uses a custom SQLite-based metadata store that replicates core MLMD functionality.

## Project Structure
```
MLMD_Labs/
├── mlmd_iris.db          # SQLite database containing metadata
├── lab_notebook.ipynb    # Main Jupyter notebook
└── README.md             # This file
```

## Implementation Details

### Core Components

1. **MLMetadataStore Class**
   - Custom implementation of MLMD functionality using SQLite
   - Manages four primary tables: artifacts, executions, events, and contexts
   - Provides methods for creating and querying metadata entities

2. **Artifacts**
   - Dataset: Iris dataset with 150 samples, 4 features, 3 classes
   - Model: RandomForest classifier with 100 estimators
   - Metrics: Evaluation metrics including accuracy, precision, recall, F1-score

3. **Executions**
   - DataIngestion: Loads and prepares the Iris dataset
   - ModelTraining: Trains the RandomForest classifier
   - ModelEvaluation: Evaluates model performance on test set

4. **Events**
   - Links artifacts to executions through INPUT and OUTPUT relationships
   - Establishes complete lineage tracking

### Pipeline Architecture
```
Data Ingestion (Execution)
        |
        v (OUTPUT)
Iris Dataset (Artifact)
        |
        v (INPUT)
Model Training (Execution)
        |
        v (OUTPUT)
Trained Model (Artifact)
        |
        v (INPUT)
Evaluation (Execution)
        |
        v (OUTPUT)
Metrics (Artifact)
```

## Methodology

### Data Processing
- Dataset: Iris flower dataset from scikit-learn
- Train-test split: 70-30 ratio with stratification
- Features: 4 numerical features (sepal length, sepal width, petal length, petal width)
- Target: 3 classes (setosa, versicolor, virginica)

### Model Configuration
- Algorithm: RandomForest Classifier
- Hyperparameters:
  - n_estimators: 100
  - max_depth: 5
  - random_state: 42

### Evaluation Metrics
- Accuracy
- Precision (weighted average)
- Recall (weighted average)
- F1-Score (weighted average)

## Results

### Model Performance
- Accuracy: 97.78%
- Precision: 97.87%
- Recall: 97.78%
- F1-Score: 97.77%

### Metadata Statistics
- Total Artifacts: 3
- Total Executions: 3
- Total Events: 6
- Total Contexts: 1

## Visualizations

The notebook includes the following visualizations:

1. **Dataset Analysis**
   - Class distribution bar chart
   - Feature distribution boxplots

2. **Model Analysis**
   - Feature importance horizontal bar chart

3. **Evaluation Results**
   - Performance metrics bar chart
   - Confusion matrix heatmap

4. **Metadata Summary**
   - MLMD store entity counts

## Key Learnings

1. **Artifact Management**: Artifacts represent immutable pieces of data such as datasets, models, and metrics.

2. **Execution Tracking**: Executions capture the computational processes that transform inputs into outputs.

3. **Lineage Establishment**: Events create relationships between artifacts and executions, enabling full traceability.

4. **Metadata Querying**: The MLMD store supports complex queries to retrieve artifacts, executions, and their relationships.

5. **Reproducibility**: Complete metadata tracking enables experiment reproducibility and model governance.

## Database Schema

### Artifacts Table
- id (INTEGER PRIMARY KEY)
- type (TEXT)
- name (TEXT)
- uri (TEXT)
- properties (TEXT/JSON)
- created_at (TEXT)

### Executions Table
- id (INTEGER PRIMARY KEY)
- type (TEXT)
- name (TEXT)
- state (TEXT)
- properties (TEXT/JSON)
- created_at (TEXT)

### Events Table
- id (INTEGER PRIMARY KEY)
- artifact_id (INTEGER FOREIGN KEY)
- execution_id (INTEGER FOREIGN KEY)
- type (TEXT)
- created_at (TEXT)

### Contexts Table
- id (INTEGER PRIMARY KEY)
- type (TEXT)
- name (TEXT)
- properties (TEXT/JSON)
- created_at (TEXT)

## Usage

To run the lab:
```bash
jupyter lab
```

Open `lab_notebook.ipynb` and execute cells sequentially from Step 1 through Step 7.

## Future Enhancements

- Integration with TensorFlow Extended (TFX) pipelines
- Support for distributed metadata stores
- Implementation of artifact versioning
- Addition of provenance tracking for data transformations
- Integration with model registry systems

## References

- ML Metadata Documentation: https://www.tensorflow.org/tfx/guide/mlmd
- Iris Dataset: Fisher, R.A. "The use of multiple measurements in taxonomic problems" (1936)
- scikit-learn Documentation: https://scikit-learn.org/

## Author

Graduate Student - Data Science Program

## License

This project is for educational purposes as part of the ML Metadata lab assignment.