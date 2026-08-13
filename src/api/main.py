from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.explainability.explain import explain_prediction
from src.monitoring.drift import run_drift_analysis

MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_THRESHOLD = 0.4

app = FastAPI(
    title="Predictive Maintenance API",
    version="0.1.0",
    description="Predict industrial machine failure risk.",
)


class MachineInput(BaseModel):
    type: str = Field(pattern="^[LMH]$")
    air_temperature: float
    process_temperature: float
    rotational_speed: int = Field(gt=0)
    torque: float = Field(ge=0)
    tool_wear: int = Field(ge=0)


class FeatureContribution(BaseModel):
    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    failure_probability: float
    predicted_failure: bool
    threshold: float
    explanation: list[FeatureContribution]


@lru_cache
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. Train the model before starting the API."
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
    }

@app.get("/monitor/drift")
def monitor_drift() -> dict:
    return run_drift_analysis()

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: MachineInput) -> PredictionResponse:
    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    row = pd.DataFrame(
        [
            {
                "Type": payload.type,
                "Air temperature": payload.air_temperature,
                "Process temperature": payload.process_temperature,
                "Rotational speed": payload.rotational_speed,
                "Torque": payload.torque,
                "Tool wear": payload.tool_wear,
            }
        ]
    )

    probability = float(model.predict_proba(row)[0, 1])
    explanation = explain_prediction(row)
    return PredictionResponse(
        failure_probability=round(probability, 6),
        predicted_failure=probability >= DEFAULT_THRESHOLD,
        threshold=DEFAULT_THRESHOLD,
        explanation=explanation,
)
