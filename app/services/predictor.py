"""Core prediction service — loads model and runs inference."""
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PredictionService:
    """Wraps the trained ML model for real-time inference."""

    MODEL_PATH = Path("ml/saved_models/best_model.joblib")
    META_PATH  = Path("ml/saved_models/model_metadata.json")

    # Hard-coded encodings (mirror of LabelEncoder fit)
    TEAM_MAP    = {"Red Bull":0,"Mercedes":1,"Ferrari":2,"McLaren":3,
                   "Aston Martin":4,"Alpine":5,"RB":6,"Haas":7,"Williams":8,"Sauber":9}
    WEATHER_MAP = {"Dry":0,"Mixed":1,"Wet":2}
    TEAM_STR    = {"Red Bull":0.95,"Mercedes":0.82,"Ferrari":0.80,"McLaren":0.78,
                   "Aston Martin":0.65,"Alpine":0.55,"RB":0.50,
                   "Haas":0.45,"Williams":0.42,"Sauber":0.38}

    def __init__(self):
        self.model    = None
        self.metadata = {}
        self._load()

    def _load(self):
        try:
            self.model    = joblib.load(self.MODEL_PATH)
            with open(self.META_PATH) as f:
                self.metadata = json.load(f)
            logger.info(f"Model loaded: {self.metadata.get('model_name')}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    def _build_features(self, req: dict) -> pd.DataFrame:
        """Convert API request dict to model feature vector."""
        team    = req.get("team", "Williams")
        weather = req.get("weather", "Dry")
        grid    = req.get("grid_position", 10)
        prev    = req.get("previous_finish", 10)
        row = {
            "grid_position":             grid,
            "q1_time_seconds":           80 + grid * 0.15,
            "q2_time_seconds":           80 + grid * 0.10 if grid <= 15 else 0,
            "q3_time_seconds":           80 + grid * 0.05 if grid <= 10 else 0,
            "pit_stop_count":            2,
            "avg_lap_time":              91.0,
            "wet_race":                  1 if weather in ["Wet","Mixed"] else 0,
            "season":                    req.get("season", 2025),
            "round_number":              req.get("round_number", 1),
            "team_encoded":              self.TEAM_MAP.get(team, 9),
            "track_encoded":             0,
            "weather_encoded":           self.WEATHER_MAP.get(weather, 0),
            "driver_code_encoded":       0,
            "driver_consistency_score":  3.5,
            "team_performance_score":    self.TEAM_STR.get(team, 0.5),
            "avg_finish_last5":          prev,
            "qualifying_delta":          grid * 0.12,
            "track_avg_finish":          prev,
            "prev_finish":               prev,
            "team_strength":             self.TEAM_STR.get(team, 0.5),
        }
        return pd.DataFrame([row])

    def predict(self, req: dict) -> dict:
        """Run inference and return prediction + probability."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        X   = self._build_features(req)
        pred = bool(self.model.predict(X)[0])
        prob = float(self.model.predict_proba(X)[0][1])
        conf = "High" if prob > 0.75 else "Medium" if prob > 0.50 else "Low"
        return {
            "driver":           req.get("driver"),
            "team":             req.get("team"),
            "track":            req.get("track"),
            "top10_prediction": pred,
            "probability":      round(prob, 4),
            "confidence":       conf,
            "model_version":    self.metadata.get("version", "v1.0.0"),
        }

    @property
    def is_loaded(self) -> bool:
        return self.model is not None