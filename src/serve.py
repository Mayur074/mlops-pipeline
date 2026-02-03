"""
src/serve.py
-------------
FastAPI inference server that loads the best registered MLflow model
and serves real-time predictions with monitoring.
"""

from __future__ import annotations
import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import List

import numpy as np
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title    = "MLOps Churn Prediction API",
    version  = "1.0.0",
    description = "Production ML inference API with MLflow model registry integration.",
)

# ── Globals ───────────────────────────────────────────────────────────────────
model          = None
prediction_log = []   # In-memory monitoring store
MLFLOW_DIR     = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
RESULTS_PATH   = "models/training_results.json"


# ── Schema ────────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="Feature vector (15 values for synthetic dataset)",
        example=[0.5] * 15,
    )

class PredictionResponse(BaseModel):
    prediction:  int
    probability: float
    label:       str
    latency_ms:  float
    timestamp:   str

class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    total_predictions: int
    avg_latency_ms: float


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
def load_model():
    global model
    mlflow.set_tracking_uri(MLFLOW_DIR)

    try:
        if Path(RESULTS_PATH).exists():
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            run_id = results["best_model"]["run_id"]
            model_uri = f"runs:/{run_id}/model"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Loaded model from run {run_id}")
        else:
            logger.warning(
                "No training results found. Run src/train.py first. "
                "API will return 503 until model is loaded."
            )
    except Exception as e:
        logger.error(f"Model loading failed: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    avg_latency = (
        sum(p["latency_ms"] for p in prediction_log) / len(prediction_log)
        if prediction_log else 0.0
    )
    return HealthResponse(
        status           = "healthy" if model else "degraded",
        model_loaded     = model is not None,
        total_predictions= len(prediction_log),
        avg_latency_ms   = round(avg_latency, 2),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run training pipeline first."
        )

    start = time.perf_counter()
    X     = np.array(request.features).reshape(1, -1)

    try:
        pred    = int(model.predict(X)[0])
        proba   = float(model.predict_proba(X)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    ts         = datetime.utcnow().isoformat()

    # Log for monitoring
    prediction_log.append({
        "prediction": pred, "probability": proba,
        "latency_ms": latency_ms, "timestamp": ts,
    })

    return PredictionResponse(
        prediction  = pred,
        probability = round(proba, 4),
        label       = "churn" if pred == 1 else "retained",
        latency_ms  = latency_ms,
        timestamp   = ts,
    )


@app.get("/metrics")
def metrics():
    """Return prediction monitoring metrics."""
    if not prediction_log:
        return {"message": "No predictions yet."}

    probs    = [p["probability"] for p in prediction_log]
    latencies= [p["latency_ms"]  for p in prediction_log]
    preds    = [p["prediction"]  for p in prediction_log]

    return {
        "total_predictions":  len(prediction_log),
        "churn_rate":         round(sum(preds) / len(preds), 4),
        "avg_probability":    round(sum(probs) / len(probs), 4),
        "avg_latency_ms":     round(sum(latencies) / len(latencies), 2),
        "max_latency_ms":     max(latencies),
        "recent_predictions": prediction_log[-5:],
    }


@app.get("/")
def root():
    return {
        "service": "MLOps Churn Prediction API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "/predict",
        "metrics": "/metrics",
    }
