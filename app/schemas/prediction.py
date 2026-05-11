"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from enum import Enum


class WeatherType(str, Enum):
    DRY   = "Dry"
    WET   = "Wet"
    MIXED = "Mixed"


class PredictionRequest(BaseModel):
    driver:          str           = Field(..., example="VER",      description="Driver 3-letter code")
    team:            str           = Field(..., example="Red Bull",  description="Constructor name")
    grid_position:   int           = Field(..., ge=1, le=20,        description="Starting grid position")
    track:           str           = Field(..., example="Monaco Grand Prix")
    weather:         WeatherType   = Field(default=WeatherType.DRY)
    previous_finish: Optional[int] = Field(default=10, ge=1, le=20)
    season:          Optional[int] = Field(default=2025)
    round_number:    Optional[int] = Field(default=1, ge=1, le=24)

    @validator("driver")
    def driver_uppercase(cls, v):
        return v.upper().strip()


class PredictionResponse(BaseModel):
    driver:           str
    team:             str
    track:            str
    top10_prediction: bool
    probability:      float = Field(..., ge=0.0, le=1.0)
    confidence:       str
    model_version:    str


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
    timestamp:     str


class ModelInfoResponse(BaseModel):
    model_name:    str
    version:       str
    trained_at:    str
    metrics:       dict
    feature_count: int