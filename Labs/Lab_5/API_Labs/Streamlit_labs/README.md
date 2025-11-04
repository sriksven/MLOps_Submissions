# Job Salary Prediction Streamlit Project

## Overview
This project builds an end to end salary prediction application based on the Data Science Job Salaries 2023 dataset. The goal is to predict salary in USD given job attributes such as job title, experience, employment type, remote ratio, company location and company size.

The application stack includes preprocessing scripts, model training pipeline, and a Streamlit UI for interactive predictions and insights.

## Project Structure

```
.
├── data
│   ├── salaries.csv
│   └── cleaned_data.csv
├── models
│   ├── salary_model.pkl
│   └── label_encoders.pkl
├── app.py (Streamlit App)
├── preprocess.py
└── train_model.py
```

## Requirements
Below libraries are used for this project.

```
streamlit==1.38.0
pandas==2.2.3
scikit_learn==1.5.2
joblib==1.4.2
xgboost==2.1.1
numpy==1.26.4
matplotlib==3.9.2
seaborn==0.13.2
shap==0.44.0
```

## Steps to Reproduce

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Preprocess Dataset

This script:
1. Loads raw CSV
2. Drops unnecessary columns
3. Encodes categorical columns using LabelEncoder
4. Saves cleaned CSV and encoders

Run
```
python preprocess.py
```

This generates
- data/cleaned_data.csv
- models/label_encoders.pkl

### 3. Train the Model

This script:
1. Loads cleaned data
2. Splits into train and test
3. Trains a Random Forest Regressor
4. Saves trained model

Run
```
python train_model.py
```

This generates
- models/salary_model.pkl

### 4. Run Streamlit App
```
streamlit run app.py
```

## App Features

- Predict salary based on selected job attributes
- Encoders map categorical text into numbers
- Visual charts showing average salaries by job title and remote ratio

## Important Notes

- Data preprocessing must be run before model training
- Encoders must be saved and loaded inside the app
- Streamlit app directly reads cleaned data and models

## Dataset Reference

Dataset used is from Kaggle
Data Science Job Salaries 2023 Dataset

## Next Improvements

- Add experiment tracking
- Add hyperparameter tuning
- Add SHAP explainability in UI

## License
Open use for educational purposes.
