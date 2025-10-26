# 🧠 CSGO ML Lab – Simple Machine Learning Project in Docker

A lightweight, end-to-end machine learning project that trains a simple classifier on **CSGO Pro Players Dataset** to identify **High Performers** based on kills, deaths, and player rating.  
Everything runs fully inside Docker — no local dependencies required.

## 🗂 Project Structure

```
Lab1/
├── data/
│   ├── csgo_players.csv         # dataset from Kaggle
│   └── csgo_model.pkl           # trained model (created after training)
│
├── src/
│   ├── train.py                 # training + cleaning pipeline
│   └── predict.py               # load model and predict new samples
│
├── Dockerfile                   # defines container environment
├── requirements.txt              # dependencies
└── README.md                    # this file
```

## 🎯 Objective

Train a small ML model that predicts whether a professional CSGO player is a **High Performer** based on:
- Total kills  
- Total deaths  
- Player rating  

The label is created by checking if `rating > 1.0`.

## 🧰 Dependencies

```
pandas
scikit-learn
joblib
```

## 📥 Dataset

Download the dataset from Kaggle:  
👉 [https://www.kaggle.com/datasets/naumanaarif/csgo-pro-players-dataset](https://www.kaggle.com/datasets/naumanaarif/csgo-pro-players-dataset)

Save it as:

```
data/csgo_players.csv
```

## ⚙️ Local Setup (Optional)

```bash
python3 -m venv venv
source venv/bin/activate     # on macOS/Linux
pip install -r requirements.txt
python src/train.py
python src/predict.py 20000 15000
```

Expected output:
```
Model Accuracy: 0.88
✅ Model saved as data/csgo_model.pkl
Prediction: High Performer
```

## 🐳 Docker Setup (Recommended)

### Build the Docker Image
```bash
docker build -t csgo-ml-lab .
```

### Train the Model
```bash
docker run -v $(pwd):/app csgo-ml-lab
```

✅ Output:
```
Model Accuracy: 0.90
✅ Model saved as data/csgo_model.pkl
```

### Predict Inside Docker
```bash
docker run -v $(pwd):/app csgo-ml-lab python src/predict.py 20000 15000
```

✅ Example Output:
```
Prediction: High Performer
```

## 🧹 Data Cleaning Summary

- Uses only: `total_kills`, `total_deaths`, and `rating`
- Converts values to numeric safely
- Drops missing or invalid rows
- Creates `KD_ratio = total_kills / (total_deaths + 1)`
- Generates label: `High_Performer = 1 if rating > 1.0 else 0`

## 🧠 Model Details

| Component | Description |
|------------|--------------|
| **Algorithm** | RandomForestClassifier |
| **Framework** | scikit-learn |
| **Features** | total_kills, total_deaths, KD_ratio |
| **Target** | High_Performer |
| **Metric** | Accuracy |

## 🧩 Example Predictions

| Input (Kills, Deaths) | Output |
|------------------------|--------|
| (25000, 18000) | High Performer |
| (10000, 25000) | Average Player |

## 🧾 Example Docker Commands

| Task | Command |
|------|----------|
| Build Image | `docker build -t csgo-ml-lab .` |
| Train Model | `docker run -v $(pwd):/app csgo-ml-lab` |
| Predict | `docker run -v $(pwd):/app csgo-ml-lab python src/predict.py 25000 18000` |
| List Images | `docker images` |
| Remove Container | `docker rm <container_id>` |

## 🧑‍💻 Author

**Sriks V**  

## 🧾 License
This project is for educational purposes only.  
Dataset copyright belongs to its original author on Kaggle.
