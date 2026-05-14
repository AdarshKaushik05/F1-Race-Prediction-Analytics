# 🏎️ F1 Race Position Prediction & Analytics System

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)](https://xgboost.ai)
[![FastF1](https://img.shields.io/badge/FastF1-3.3-red)](https://docs.fastf1.dev)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Production-grade ML system** that predicts whether an F1 driver will finish in the **Top 10**
> using **real Formula 1 telemetry data** via the FastF1 library — covering qualifying times,
> pit stops, lap times, weather conditions, and race results across multiple seasons.

<div align="center">
**🚀 [Click here to launch the Live Web Application!](https://f1-race-prediction-analytics-gbhmwgmmai3csko5lc7hbj.streamlit.app/)**
</div>

<div align="center">
  <h2>📸 System Snapshots</h2>

  <h3>Dashboard Home</h3>
  <img src="https://github.com/user-attachments/assets/6ce316f5-2b6f-4d31-b3a4-2f75d1cc6f16" width="800" alt="Dashboard Home" />
  <br><br>

  <h3>Prediction Engine</h3>
  <img src="https://github.com/user-attachments/assets/e65c835f-7bee-45e3-a567-887b5e0e0ddd" width="800" alt="Prediction" />
  <br><br>

  <h3>Driver Analytics</h3>
  <img src="https://github.com/user-attachments/assets/2a2fcf52-2227-4165-9407-3b8e97554320" width="800" alt="Driver Analytics" />
</div>
 

---

## ✨ Features

- 🎯 **Binary Classification** — Top-10 finish prediction with real-time probability score
- 📡 **Real F1 Data** — FastF1 library pulls official telemetry, qualifying, pit stops & weather
- 🤖 **3 ML Models** — Logistic Regression, Random Forest, XGBoost (best: 90.6% ROC-AUC)
- 🔍 **Explainable AI** — SHAP values, feature importance, waterfall plots per prediction
- 📊 **Interactive Dashboard** — Streamlit with Plotly charts, dark F1 theme
- ⚡ **REST API** — FastAPI with Swagger UI, Pydantic validation, async support
- 🗓️ **Multi-Season** — 2018–2024 training data (1,559 records), 2025 test data (259 records)
- 🌤️ **Real Weather** — Dry (86.8%), Mixed (12.1%), Wet (1.1%) from F1 sensor data
- 🐳 **Docker Ready** — Multi-stage build for API + Dashboard

---

## 🔄 Synthetic → FastF1: What Changed

This project was upgraded from a synthetic data generator to **real F1 data via FastF1**.
Here is exactly what changed and why it matters:

| Property | Synthetic (old) | FastF1 (current) |
|---|---|---|
| **Data source** | Statistical distributions | Official F1 timing server |
| **Total records** | 3,680 (7 seasons × 23 rounds × 20 drivers) | 1,818 (real races, some seasons partial) |
| **Train records** | 3,220 | 1,559 |
| **Test records** | 460 | 259 |
| **Qualifying times** | `80 + grid × 0.15` (estimated) | Real Q1 / Q2 / Q3 lap times in seconds |
| **Pit stop count** | `random.choice([1,2,3])` | Actual count from lap data |
| **Avg lap time** | `91 + noise` | Median of clean laps (pit/safety car excluded) |
| **Weather** | 75% Dry, 15% Wet, 10% Mixed (random) | F1 Rainfall sensor: 86.8% Dry, 12.1% Mixed, 1.1% Wet |
| **Team strength** | Hard-coded dict (Red Bull=0.95 etc.) | Derived from actual top-10 finish rates |
| **DNF rate** | 12% random | Actual race retirements per status field |
| **Unique drivers** | 20 (current grid only) | 40 (covers driver changes 2018–2025) |
| **Unique tracks** | 23 | 29 (includes historical venue variations) |
| **Team names** | Simplified (e.g. "Red Bull") | Official FastF1 names (e.g. "Red Bull Racing") |
| **Top-10 rate** | Exactly 50.0% (forced) | 48.5% (realistic — accounts for DNFs) |
| **Best model** | XGBoost (synthetic over-fit) | Logistic Regression (ROC-AUC 0.9059) |
| **ALB P18 Wet prediction** | 100% ✅ (unrealistic) | 0.4% ❌ (realistic — Williams from 18th in wet) |
| **MAG P16 Dry prediction** | 100% ✅ (unrealistic) | 1.8% ❌ (realistic — Haas from 16th) |

> **Key insight:** With synthetic data, every driver was predicted Top 10 at ~100% because the
> data had no real signal — team strength was uniform noise. With FastF1, the model correctly
> distinguishes that a Williams starting 18th in the wet has a 0.4% chance while a Red Bull
> starting on pole has a 99.9% chance.

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
│   LogisticRegression ⭐ │ RF │ XGBoost      │
│   Feature Engineering Pipeline             │
│   SHAP Explainer                           │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│              Data Layer                     │
│         FastF1 Official F1 Telemetry        │
│   Race Results · Qualifying · Pit Stops     │
│   Lap Times · Weather Sensor Data           │
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
│   └── main.py          # FastAPI app entry point
├── ml/
│   ├── data/
│   │   ├── raw/         # f1_raw_data.csv (FastF1 output)
│   │   └── processed/   # train_processed.csv, test_processed.csv
│   ├── preprocessing/   # F1FeatureEngineer
│   ├── training/        # F1ModelTrainer
│   ├── evaluation/      # Metrics + SHAP
│   └── saved_models/    # best_model.joblib + model_metadata.json
├── streamlit_app/
│   └── app.py           # Streamlit 5-page dashboard
├── tests/               # pytest unit tests
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .streamlit/config.toml
└── README.md
```

---

## 🚀 Quick Start

### Option A — Google Colab (Recommended)

1. Open `F1_Race_Predictor.ipynb` in [Google Colab](https://colab.research.google.com)
2. Run Sections 1 & 2 from the main notebook
3. Run the `F1_FastF1_DataCollection_PATCH.ipynb` for data collection
4. Continue from Section 4 onward — everything runs unchanged

> **Tip:** Mount Google Drive to cache FastF1 data and avoid re-downloading on each session.

### Option B — Local

```bash
git clone https://github.com/AdarshKaushik05/F1-Race-Prediction-Analytics
cd F1-Race-Prediction-Analytics
pip install -r requirements.txt

# Start API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start Dashboard (new terminal)
streamlit run streamlit_app/app.py
```

### Option C — Docker

```bash
docker-compose up --build
# API       → http://localhost:8000
# Dashboard → http://localhost:8501
# Swagger   → http://localhost:8000/docs
```

---

## 📡 API Usage

```bash
# Health Check
curl http://localhost:8000/

# Top-10 Prediction
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "driver": "VER",
    "team": "Red Bull Racing",
    "grid_position": 1,
    "track": "Monaco Grand Prix",
    "weather": "Dry",
    "previous_finish": 1
  }'

# Response
# {"top10_prediction": true, "probability": 0.999, "confidence": "High"}

# List endpoints
curl http://localhost:8000/drivers
curl http://localhost:8000/teams
curl http://localhost:8000/tracks
curl http://localhost:8000/model-info
```

Full interactive docs: `http://localhost:8000/docs`

---

## 📊 Model Performance (Real FastF1 Data)

Trained on **1,559 records** from 2018–2024 FastF1 telemetry data.
Validated on a held-out 20% split (312 records).

| Model | CV ROC-AUC | Val ROC-AUC | F1 | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.9320 ± 0.016 | **0.9059** | **0.8082** | **82.1%** |
| Random Forest | 0.9247 ± 0.018 | 0.8995 | 0.7919 | 80.1% |
| XGBoost | 0.9282 ± 0.013 | 0.8873 | 0.7692 | 77.9% |

```
Classification Report — Logistic Regression (best model):

                precision  recall  f1-score  support
Outside Top 10    0.80      0.87     0.83      159
Top 10            0.85      0.77     0.81      153

accuracy                             0.82      312
```

> **Why Logistic Regression wins on real data:** With real F1 data the features
> (grid position, qualifying delta, team strength) have strong linear relationships
> with Top-10 outcomes. Tree-based models over-fit the smaller real dataset (1,559
> records vs 3,220 synthetic). LR generalises better with regularisation.

---

## 🔍 Key Engineered Features

| Feature | Description | Source |
|---|---|---|
| `qualifying_delta` | Gap to pole lap time (seconds) | FastF1 qualifying session |
| `avg_finish_last5` | Driver's avg finish over last 5 races | Rolling window on race results |
| `driver_consistency_score` | Std deviation of last-5 finishes (lower = better) | Rolling window |
| `team_performance_score` | Team's rolling avg points normalised | Race results |
| `track_avg_finish` | Driver's historical avg at this specific track | Grouped by driver+track |
| `wet_race` | Binary flag from F1 Rainfall sensor | FastF1 weather data |
| `pit_stop_count` | Actual stops from lap PitOutTime | FastF1 lap data |
| `avg_lap_time` | Median clean lap time (excl. pit/SC laps) | FastF1 lap data |

---

## 🌤️ Real Weather Distribution (FastF1)

```
Dry     : 1,578 races  (86.8%)   ████████████████████████████████████████
Mixed   :   220 races  (12.1%)   ████████████████████████
Wet     :    20 races  ( 1.1%)   ██
```

---

## 🏢 Team Strengths (Derived from Real Results)

```
Ferrari              0.84  ████████████████
Mercedes             0.83  ████████████████
Red Bull Racing      0.79  ███████████████
McLaren              0.73  ██████████████
Red Bull             0.71  ██████████████
Renault              0.59  ███████████
Force India          0.56  ███████████
Aston Martin         0.44  ████████
Alpine               0.39  ███████
AlphaTauri           0.31  ██████
Haas F1 Team         0.20  ████
Williams             0.19  ████
```

---

## 🗓️ Data Coverage

```
Season  Records  Races  Notes
──────────────────────────────────────────────
2018     340      17    Partial (API rate limit)
2021     300      15    Partial (API rate limit)
2023     440      22    Full season
2024     479      24    Full season
2025     259      13    Partial (ongoing season)
──────────────────────────────────────────────
TRAIN   1,559     (2018–2024)
TEST      259     (2025)
TOTAL   1,818     40 drivers · 29 tracks
```

---

## 🔮 Future Improvements

- [ ] Full season coverage for 2018–2022 (requires longer FastF1 fetch time)
- [ ] Live race weekend predictions using real-time FastF1 telemetry
- [ ] Monte Carlo race simulation engine
- [ ] LLM-generated race commentary (GPT/Claude integration)
- [ ] Driver clustering — unsupervised analysis of driving styles
- [ ] MLflow experiment tracking and model registry
- [ ] PostgreSQL storage for prediction history
- [ ] GitHub Actions CI/CD with automated retraining each race weekend

---

## 👨‍💻 Author

**Adarsh Kaushik**
GitHub: [@AdarshKaushik05](https://github.com/AdarshKaushik05)

---

*Built as a professional ML engineering portfolio project using real Formula 1 telemetry data.*
