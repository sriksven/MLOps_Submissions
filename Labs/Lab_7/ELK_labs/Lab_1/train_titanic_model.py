import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

logging.info("Dataset loaded successfully.")
logging.info(f"Shape: {data.shape}")

# Preprocessing
data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
data.dropna(inplace=True)
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logging.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train model
model = LogisticRegression(max_iter=500)
logging.info("Starting training...")
model.fit(X_train, y_train)
logging.info("Training complete.")

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

logging.info(f"Accuracy: {acc:.2f}")
logging.info(f"F1 Score: {f1:.2f}")
logging.info(f"Confusion Matrix: {conf_matrix.tolist()}")

# Compute TP, TN, FP, FN
tp = conf_matrix[1, 1]
tn = conf_matrix[0, 0]
fp = conf_matrix[0, 1]
fn = conf_matrix[1, 0]

logging.info(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
logging.info(f"Model coefficients: {model.coef_.tolist()}")
logging.info(f"Model intercept: {model.intercept_.tolist()}")

logging.info("Model evaluation completed.")
