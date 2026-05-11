"""FastAPI application entry point."""
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    HealthResponse, ModelInfoResponse
)
from app.services.predictor import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🏎️ F1 Race Predictor API",
    description="Predict whether an F1 driver will finish in the Top 10",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

svc = PredictionService()

DRIVERS = ["VER","HAM","LEC","PER","SAI","NOR","RUS","ALO","STR","PIA",
           "GAS","OCO","TSU","RIC","MAG","HUL","BOT","ZHO","ALB","SAR"]
TEAMS   = ["Red Bull","Mercedes","Ferrari","McLaren","Aston Martin",
           "Alpine","RB","Haas","Williams","Sauber"]
TRACKS  = ["Bahrain Grand Prix","Saudi Arabian Grand Prix","Australian Grand Prix",
           "Japanese Grand Prix","Miami Grand Prix","Monaco Grand Prix",
           "Canadian Grand Prix","Spanish Grand Prix","British Grand Prix",
           "Hungarian Grand Prix","Belgian Grand Prix","Dutch Grand Prix",
           "Italian Grand Prix","Singapore Grand Prix","United States Grand Prix",
           "Mexico City Grand Prix","São Paulo Grand Prix","Abu Dhabi Grand Prix"]


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=svc.is_loaded,
        model_version=svc.metadata.get("version", "unknown"),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    try:
        result = svc.predict(request.dict())
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/drivers", tags=["Reference"])
async def list_drivers():
    return {"drivers": DRIVERS, "count": len(DRIVERS)}


@app.get("/teams", tags=["Reference"])
async def list_teams():
    return {"teams": TEAMS, "count": len(TEAMS)}


@app.get("/tracks", tags=["Reference"])
async def list_tracks():
    return {"tracks": TRACKS, "count": len(TRACKS)}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if not svc.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        model_name=svc.metadata.get("model_name", "unknown"),
        version=svc.metadata.get("version", "unknown"),
        trained_at=svc.metadata.get("trained_at", ""),
        metrics=svc.metadata.get("metrics", {}),
        feature_count=len(svc.metadata.get("feature_names", []))
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)