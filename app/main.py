"""FastAPI churn scoring service.

Endpoints:
  GET  /health         -> liveness + model-loaded status
  POST /predict        -> score one customer
  POST /batch_predict  -> score many customers

Loads the model produced by train_model.py (model.pkl). Returns a churn probability, a predicted class
at the model's recall-leaning threshold, a coarse risk level, and a short rule-based risk explanation
derived from the customer's most churn-indicative features.
"""
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (CustomerFeatures, PredictionResponse, BatchPredictRequest,
                         BatchPredictResponse, HealthResponse)

MODEL_PATH = Path(__file__).parent.parent / "model.pkl"

# Holds the loaded artifact: {"model", "features", "numeric", "categorical", "threshold"}
STATE: dict = {}


def load_model() -> None:
    if MODEL_PATH.exists():
        STATE.update(joblib.load(MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    STATE.clear()


app = FastAPI(
    title="D2C Churn Scoring API",
    description="Scores a customer's 60-day churn risk for the retention team (decision-support only).",
    version="1.0.0",
    lifespan=lifespan,
)

# Also load eagerly at import so the model is available under TestClient (which does not
# trigger lifespan for a bare module-level client) and any non-ASGI import of `app`.
load_model()


def _risk_level(p: float) -> str:
    if p >= 0.66:
        return "high"
    if p >= 0.33:
        return "medium"
    return "low"


def _explain(feat: CustomerFeatures, proba: float) -> str:
    """Rule-based explanation from the top churn drivers (recency, web inactivity, engagement, sentiment)."""
    reasons = []
    if feat.recency_days >= 90:
        reasons.append(f"no purchase in {feat.recency_days} days")
    if feat.last_visit_days_ago >= 21:
        reasons.append(f"last app/web visit {feat.last_visit_days_ago} days ago")
    if feat.frequency_180d <= 1:
        reasons.append("low purchase frequency")
    if feat.category_diversity_180d <= 1:
        reasons.append("narrow category breadth")
    if feat.ticket_count_90d > 0 and feat.negative_ticket_rate_90d >= 0.5:
        reasons.append("recent negative support sentiment")
    if feat.return_rate_180d >= 0.5:
        reasons.append("high return rate")

    if proba >= 0.66:
        if reasons:
            return "Elevated churn risk: " + ", ".join(reasons) + "."
        return "Elevated churn risk based on the combined behavioural profile."
    if proba >= 0.33:
        if reasons:
            return "Moderate churn risk: " + ", ".join(reasons) + "."
        return "Moderate churn risk; monitor engagement."
    # low
    protective = []
    if feat.recency_days <= 30:
        protective.append("recent purchase")
    if feat.frequency_180d >= 3:
        protective.append("frequent buyer")
    if feat.last_visit_days_ago <= 7:
        protective.append("recently active")
    if protective:
        return "Low churn risk: " + ", ".join(protective) + "."
    return "Low churn risk based on current activity."


def _score_one(feat: CustomerFeatures) -> PredictionResponse:
    if "model" not in STATE:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py to create model.pkl.")
    row = pd.DataFrame([feat.model_dump()])[STATE["features"]]
    proba = float(STATE["model"].predict_proba(row)[:, 1][0])
    pred = int(proba >= STATE["threshold"])
    return PredictionResponse(
        churn_probability=round(proba, 4),
        predicted_class=pred,
        risk_level=_risk_level(proba),
        risk_explanation=_explain(feat, proba),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded="model" in STATE)


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    return _score_one(customer)


@app.post("/batch_predict", response_model=BatchPredictResponse)
def batch_predict(request: BatchPredictRequest) -> BatchPredictResponse:
    preds = [_score_one(c) for c in request.customers]
    return BatchPredictResponse(count=len(preds), predictions=preds)
