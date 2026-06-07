"""API test cases. Run with: pytest

Covers the three required scenarios:
  1. /health returns 200 and status ok
  2. /predict returns a valid, well-formed prediction for a good payload
  3. /predict rejects an invalid payload with 422 (input validation)
Plus: a high-risk customer scores higher than a low-risk one, and /batch_predict works.

The tests train the model on import if model.pkl is missing, so the suite is self-contained.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
if not (ROOT / "model.pkl").exists():
    import train_model
    train_model.main()

from app.main import app  # noqa: E402

client = TestClient(app)

# A healthy, recently-active, frequent buyer -> expect low risk.
LOW_RISK = {
    "city_tier": "Tier 1", "age_group": "25-34", "acquisition_channel": "Referral",
    "preferred_category": "Skin Care", "marketing_consent": "Yes", "loyalty_tier": "Gold",
    "recency_days": 5, "frequency_180d": 6, "monetary_180d": 4200.0,
    "return_rate_180d": 0.0, "avg_discount_pct_180d": 0.10, "avg_rating_180d": 4.5,
    "category_diversity_180d": 4, "ticket_count_90d": 0, "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0, "days_since_signup": 600, "sessions_30d": 12,
    "product_views_30d": 30, "cart_adds_30d": 5, "wishlist_adds_30d": 3,
    "abandoned_carts_30d": 0, "email_opens_30d": 8, "campaign_clicks_30d": 4,
    "last_visit_days_ago": 2,
}

# A long-inactive, one-off, unhappy customer -> expect high risk.
HIGH_RISK = {
    "city_tier": "Tier 3", "age_group": "45+", "acquisition_channel": "Marketplace",
    "preferred_category": "Fragrance", "marketing_consent": "No", "loyalty_tier": None,
    "recency_days": 200, "frequency_180d": 0, "monetary_180d": 0.0,
    "return_rate_180d": 0.5, "avg_discount_pct_180d": 0.05, "avg_rating_180d": None,
    "category_diversity_180d": 0, "ticket_count_90d": 3, "negative_ticket_rate_90d": 1.0,
    "avg_resolution_hours_90d": 60.0, "days_since_signup": 700, "sessions_30d": 0,
    "product_views_30d": 0, "cart_adds_30d": 0, "wishlist_adds_30d": 0,
    "abandoned_carts_30d": 0, "email_opens_30d": 0, "campaign_clicks_30d": 0,
    "last_visit_days_ago": 45,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload():
    r = client.post("/predict", json=LOW_RISK)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"churn_probability", "predicted_class", "risk_level", "risk_explanation"}
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_class"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")
    assert isinstance(body["risk_explanation"], str) and body["risk_explanation"]


def test_predict_invalid_payload_returns_422():
    bad = dict(LOW_RISK)
    bad["return_rate_180d"] = 5.0       # out of [0, 1] bound
    bad["recency_days"] = -10           # negative
    bad["city_tier"] = "Tier 9"         # not an allowed category
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_predict_missing_field_returns_422():
    incomplete = dict(LOW_RISK)
    del incomplete["recency_days"]
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422


def test_high_risk_scores_higher_than_low_risk():
    low = client.post("/predict", json=LOW_RISK).json()
    high = client.post("/predict", json=HIGH_RISK).json()
    assert high["churn_probability"] > low["churn_probability"]
    assert high["risk_level"] == "high"
    assert low["risk_level"] == "low"


def test_batch_predict():
    r = client.post("/batch_predict", json={"customers": [LOW_RISK, HIGH_RISK]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert len(body["predictions"]) == 2
    assert all(0.0 <= p["churn_probability"] <= 1.0 for p in body["predictions"])
