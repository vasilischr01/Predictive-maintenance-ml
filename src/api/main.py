from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    status,
)
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from src.explainability.explain import explain_prediction
from src.monitoring.drift import run_drift_analysis

logger = logging.getLogger(__name__)

MODEL_PATH = Path("artifacts/model.joblib")
THRESHOLD_PATH = Path(
    "artifacts/selected_threshold.json"
)

MAX_REQUEST_BYTES = 64 * 1024

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60

_request_history: dict[
    str,
    deque[float],
] = defaultdict(deque)


def _add_security_headers(
    response,
):
    response.headers[
        "X-Content-Type-Options"
    ] = "nosniff"

    response.headers[
        "X-Frame-Options"
    ] = "DENY"

    response.headers[
        "Referrer-Policy"
    ] = "no-referrer"

    response.headers[
        "Permissions-Policy"
    ] = (
        "camera=(), "
        "microphone=(), "
        "geolocation=()"
    )

    response.headers[
        "Cache-Control"
    ] = "no-store"

    return response


class SecurityMiddleware(
    BaseHTTPMiddleware
):
    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        content_length = (
            request.headers.get(
                "content-length"
            )
        )

        if content_length is not None:
            try:
                request_size = int(
                    content_length
                )

            except ValueError:
                response = JSONResponse(
                    status_code=400,
                    content={
                        "detail": (
                            "Invalid Content-Length "
                            "header"
                        ),
                    },
                )

                return _add_security_headers(
                    response
                )

            if request_size > MAX_REQUEST_BYTES:
                response = JSONResponse(
                    status_code=413,
                    content={
                        "detail": (
                            "Request body too large"
                        ),
                    },
                )

                return _add_security_headers(
                    response
                )

        client_host = (
            request.client.host
            if request.client
            else "unknown"
        )

        now = time.monotonic()

        history = _request_history[
            client_host
        ]

        while (
            history
            and now - history[0]
            >= RATE_LIMIT_WINDOW_SECONDS
        ):
            history.popleft()

        if (
            len(history)
            >= RATE_LIMIT_REQUESTS
        ):
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Too many requests"
                    ),
                },
                headers={
                    "Retry-After": str(
                        RATE_LIMIT_WINDOW_SECONDS
                    ),
                },
            )

            return _add_security_headers(
                response
            )

        history.append(
            now
        )

        response = await call_next(
            request
        )

        return _add_security_headers(
            response
        )


app = FastAPI(
    title="Predictive Maintenance API",
    version="0.3.0",
    description=(
        "Predict industrial machine failure "
        "risk using a validation-selected "
        "operating threshold."
    ),
)

app.add_middleware(
    SecurityMiddleware
)


class MachineInput(BaseModel):
    type: str = Field(
        pattern="^[LMH]$"
    )

    air_temperature: float
    process_temperature: float

    rotational_speed: int = Field(
        gt=0
    )

    torque: float = Field(
        ge=0
    )

    tool_wear: int = Field(
        ge=0
    )


class FeatureContribution(
    BaseModel
):
    feature: str
    shap_value: float


class PredictionResponse(
    BaseModel
):
    failure_probability: float
    predicted_failure: bool
    threshold: float
    explanation: list[
        FeatureContribution
    ]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifact not found."
        )

    return joblib.load(
        MODEL_PATH
    )


@lru_cache(maxsize=1)
def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            "Threshold artifact not found."
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
            "selected_threshold is "
            "missing from the threshold "
            "artifact."
        )

    threshold = float(
        threshold
    )

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
        "model_available": (
            MODEL_PATH.exists()
        ),
        "threshold_available": (
            THRESHOLD_PATH.exists()
        ),
    }


@app.get("/monitor/drift")
def monitor_drift() -> dict:
    try:
        return run_drift_analysis()

    except FileNotFoundError as exc:
        logger.warning(
            "drift_artifact_missing",
            exc_info=exc,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Drift analysis artifacts "
                "are unavailable."
            ),
        ) from exc

    except Exception as exc:
        logger.exception(
            "drift_analysis_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Drift analysis failed."
            ),
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

    except (
        FileNotFoundError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        logger.exception(
            "model_artifact_unavailable"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Model artifacts are "
                "unavailable."
            ),
        ) from exc

    row = build_model_input(
        payload
    )

    try:
        probability = float(
            model.predict_proba(
                row
            )[0, 1]
        )

    except Exception as exc:
        logger.exception(
            "prediction_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Prediction failed."
            ),
        ) from exc

    try:
        explanation = (
            explain_prediction(
                row
            )
        )

    except Exception as exc:
        logger.exception(
            "shap_explanation_failed"
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Prediction explanation "
                "failed."
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