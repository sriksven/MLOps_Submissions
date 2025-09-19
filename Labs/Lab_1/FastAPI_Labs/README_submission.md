# FastAPI Lab – Iris Classifier API

This project serves a trained Decision Tree Classifier on the Iris dataset as a REST API using **FastAPI** and **Uvicorn**.

## 🚀 Setup

```bash
# Clone repo
git clone <your-repo-url>
cd FastAPI_Labs

# Create env
pyenv virtualenv 3.12.9 mllabs
pyenv activate mllabs

# Install deps
pip install -r requirements.txt
```

Running the API

```bash
cd src
uvicorn main:app --reload
```

Swagger UI: http://127.0.0.1:8000/docs

Endpoints

Health check
GET / → { "status": "healthy" }

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

Project Structre:
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






