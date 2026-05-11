# 🏎️ F1 Race Position Prediction & Analytics System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.ai)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Production-grade ML system** that predicts whether an F1 driver will finish in the **Top 10**
> using historical race data, qualifying performance, driver/team statistics, and race conditions.

---

## 📸 Screenshots

| Dashboard Home | Prediction | Driver Analytics |
|---|---|---|
| ![home](docs/home.png) | ![predict](docs/predict.png) | ![driver](docs/driver.png) |

---

## ✨ Features

- 🎯 **Binary Classification** — Top-10 finish prediction with probability score
- 🤖 **3 ML Models** — Logistic Regression, Random Forest, XGBoost (best: ~91% ROC-AUC)
- 🔍 **Explainable AI** — SHAP values, feature importance, waterfall plots
- 📊 **Interactive Dashboard** — Streamlit with Plotly charts, dark F1 theme
- ⚡ **REST API** — FastAPI with Swagger UI, Pydantic validation, async support
- 🗓️ **7 Seasons** — 2018–2024 training data, 2025 test data
- 🐳 **Docker Ready** — Multi-stage build for API + Dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit Dashboard               │
│   (Home / Predict / Analytics / Insights)   │
└──────────────────┬──────────────────────────┘
                   │ HTTP
┌──────────────────▼──────────────────────────┐
│              FastAPI Backend                │
│   /predict  /drivers  /teams  /model-info   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           ML Prediction Service             │
│   XGBoost (best) │ RF │ LogReg              │
│   Feature Engineering Pipeline             │
│   SHAP Explainer                           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Data Layer                     │
│   Ergast API  │  FastF1  │  Synthetic Gen   │
└─────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
f1-race-predictor/
├── app/
│   ├── api/             # API route handlers
│   ├── services/        # PredictionService
│   ├── schemas/         # Pydantic models
│   └── main.py          # FastAPI app
├── ml/
│   ├── data/            # raw + processed CSVs
│   ├── preprocessing/   # FeatureEngineer
│   ├── training/        # ModelTrainer
│   ├── evaluation/      # Metrics + SHAP
│   └── saved_models/    # joblib + metadata
├── streamlit_app/
│   └── app.py           # Streamlit dashboard
├── tests/               # pytest unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended)
Open `F1_Race_Predictor.ipynb` in Google Colab and run all cells.

### Option B — Local

```bash
git clone https://github.com/yourname/f1-race-predictor
cd f1-race-predictor
pip install -r requirements.txt

# Train model
python ml/training/train.py

# Start API
uvicorn app.main:app --reload

# Start Dashboard
streamlit run streamlit_app/app.py
```

### Option C — Docker

```bash
docker-compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📡 API Usage

```bash
# Health Check
curl http://localhost:8000/

# Predict
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "driver": "VER",
    "team": "Red Bull",
    "grid_position": 1,
    "track": "Monaco Grand Prix",
    "weather": "Dry",
    "previous_finish": 1
  }'

# Response
# {"top10_prediction": true, "probability": 0.94, "confidence": "High"}
```

Swagger UI: `http://localhost:8000/docs`

---

## 📊 Model Performance

| Model | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.811 | 0.772 | 78.2% |
| Random Forest | 0.891 | 0.845 | 85.6% |
| **XGBoost** ✅ | **0.912** | **0.871** | **87.9%** |

---

## 🔮 Future Improvements

- [ ] Live telemetry ingestion via FastF1 during race weekends
- [ ] Monte Carlo race simulation engine
- [ ] LLM-generated race commentary (GPT/Claude integration)
- [ ] Driver clustering (unsupervised ML)
- [ ] MLflow experiment tracking
- [ ] PostgreSQL for historical predictions storage
- [ ] GitHub Actions CI/CD pipeline

---

*Built as a professional ML engineering portfolio project.*
