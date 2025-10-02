# FastAPI Lab - Iris Classifier API


## 🔄 Enhancements Over Original Lab

When I first started with the lab, the FastAPI application was very minimal; it only returned an integer class label inside a basic response model. The /predict endpoint simply gave me something like { "response": 2 }, and there was no way to retrain the model without running train.py manually. Also, predictions weren’t being tracked anywhere, and error handling was limited. It was a good starting point, but I wanted to push it further and make it more realistic.

So, I made several enhancements to improve both functionality and usability:

I updated the /predict endpoint to return not just the class label but also the probability scores, so I can see how confident the model is for each prediction.

I built a logging system that automatically appends every prediction into prediction_log.csv with timestamp, input features, predicted class, and probabilities. This way, I can audit or review past predictions anytime.

I added a new /retrain endpoint, which triggers train.py via subprocess, allowing me to retrain the model directly from the API instead of running scripts manually.

I also improved error handling by wrapping the logic in try/except blocks and raising HTTPException when something goes wrong, so that API users get clear error messages instead of raw crashes.

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







