# FastAPI Lab – Iris Classifier API


## 🔄 Enhancements Over Original Lab

In the original lab, the FastAPI application was limited to serving predictions by returning only a single integer class wrapped in a simple response model. There was no mechanism to track predictions, no option to retrain the model without manually running `train.py`, and error handling was minimal. The `/predict` endpoint output looked like `{ "response": 2 }`, which conveyed only the predicted class label. While this worked for a basic demonstration, it lacked features that would make the service more practical, transparent, and closer to production standards.

In the enhanced version, several key improvements were introduced:
- The `/predict` endpoint was modified to return both the predicted class and the associated probability scores, giving richer insight into the model’s confidence for each prediction.  
- A logging system was added that records every prediction request into `prediction_log.csv` along with timestamp, input features, predicted class, and probability scores — this enables auditing and reproducibility.  
- A new `/retrain` endpoint was implemented, allowing the model to be retrained directly from the API by invoking `train.py` with `subprocess`, instead of requiring manual execution.  
- Error handling was strengthened by wrapping predictions in `try/except` blocks and raising `HTTPException` when issues occur, ensuring that the API provides clear error messages.  
Together, these changes make the project more unique, robust, and aligned with real-world MLOps practices compared to the original lab.

This project serves a trained Decision Tree Classifier on the Iris dataset as a REST API using **FastAPI** and **Uvicorn**.

## Setup

```bash
# Clone repo
git clone <your-repo-url>
cd FastAPI_Labs

# Create env
pyenv virtualenv 3.12.9 mllabs
pyenv activate mllabs

# Install dependencies
pip install -r requirements.txt
```

Running the API

cd src
uvicorn main:app --reload


Swagger UI: http://127.0.0.1:8000/docs


Endpoints

Health check
- GET / → { "status": "healthy" }

Predict
POST /predict
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"petal_length":5.1,"sepal_length":3.5,"petal_width":1.4,"sepal_width":0.2}'
```

Response:
```bash
{
  "class": 2,
  "probabilities": [0.0, 0.3333, 0.6666]
}
```

Retrain
POST /retrain → retrains the model and updates iris_model.pkl

Project Structure:

```
FastAPI_Labs/
│── assets/
│── model/
│   ├── iris_model.pkl
│   └── prediction_log.csv
│── src/
│   ├── main.py
│   ├── predict.py
│   ├── train.py
│   └── data.py
│── requirements.txt
│── .gitignore
│── README_submission.md
└── README.md

```







