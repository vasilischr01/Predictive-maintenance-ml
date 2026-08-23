from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability.explain import explain_prediction
from src.monitoring.drift import run_drift_analysis


MODEL_PATH = Path("artifacts/model.joblib")
THRESHOLD_PATH = Path("artifacts/selected_threshold.json")


app = FastAPI(
    title="Predictive Maintenance API",
    version="0.2.0",
    description=(
        "Predict industrial machine failure risk using a "
        "validation-selected operating threshold."
    ),
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


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found. "
            "Train the model before starting the API."
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "Threshold artifact not found. "
            "Train the model before starting the API."
        )

    payload = json.loads(
        THRESHOLD_PATH.read_text(
            encoding="utf-8",
        )
    )

    threshold = payload.get(
        "selected_threshold"
    )

    if threshold is None:
        raise ValueError(
            "selected_threshold is missing from "
            "the threshold artifact."
        )

    threshold = float(threshold)

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "selected_threshold must be "
            "between 0 and 1."
        )

    return threshold


def build_model_input(
    payload: MachineInput,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Type": payload.type,
                "Air temperature": (
                    payload.air_temperature
                ),
                "Process temperature": (
                    payload.process_temperature
                ),
                "Rotational speed": float(
                    payload.rotational_speed
                ),
                "Torque": payload.torque,
                "Tool wear": float(
                    payload.tool_wear
                ),
            }
        ]
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists(),
        "threshold_available": (
            THRESHOLD_PATH.exists()
        ),
    }


@app.get("/monitor/drift")
def monitor_drift() -> dict:
    try:
        return run_drift_analysis()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: MachineInput,
) -> PredictionResponse:
    try:
        model = load_model()
        threshold = load_threshold()
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    row = build_model_input(payload)

    probability = float(
        model.predict_proba(row)[0, 1]
    )

    try:
        explanation = explain_prediction(
            row
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction succeeded but "
                "SHAP explanation failed: "
                f"{exc}"
            ),
        ) from exc

    return PredictionResponse(
        failure_probability=round(
            probability,
            6,
        ),
        predicted_failure=(
            probability >= threshold
        ),
        threshold=round(
            threshold,
            4,
        ),
        explanation=explanation,
    )