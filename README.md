# Part 4 — FastAPI Churn Scoring Service & Reproducible ML Workflow

**D2C Customer Churn Intelligence & Retention** · Capstone Part 4 of 4

A small internal service that turns the Part 3 churn model into an HTTP API the retention/CRM team can call.
Given a customer's snapshot features it returns a churn probability, a predicted class, a risk level, and a
short risk explanation.

## Repository structure

```
iitp_aiml_2506164_capstone_pt_4/
├── app/
│   ├── main.py            # FastAPI app: /health, /predict, /batch_predict
│   └── schemas.py         # Pydantic request/response models (input validation)
├── train_model.py         # Trains & saves model.pkl from data/ (run once before serving)
├── tests/
│   └── test_api.py        # pytest suite (health, valid predict, 422, ranking, batch)
├── data/                  # The 7 source CSVs + DATA_DICTIONARY.md (for reproducible training)
├── monitoring_plan.md     # Drift / prediction / outcome / error monitoring + responsible-use note
├── Dockerfile             # Optional containerised, self-contained build
├── requirements.txt
└── README.md
```

## Setup & run

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train_model.py                 # creates model.pkl from data/ (validation ROC-AUC ~0.886)
uvicorn app.main:app --reload         # serves on http://127.0.0.1:8000
```

Interactive docs at `http://127.0.0.1:8000/docs`.

### With Docker (optional)

```bash
docker build -t d2c-churn-api .       # trains the model during the build
docker run -p 8000:8000 d2c-churn-api
```

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + whether the model is loaded |
| POST | `/predict` | Score one customer |
| POST | `/batch_predict` | Score a list of customers (1–1000) |

### Sample request — `POST /predict`

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "city_tier": "Tier 3", "age_group": "45+", "acquisition_channel": "Marketplace",
  "preferred_category": "Fragrance", "marketing_consent": "No", "loyalty_tier": null,
  "recency_days": 200, "frequency_180d": 0, "monetary_180d": 0.0,
  "return_rate_180d": 0.5, "avg_discount_pct_180d": 0.05, "avg_rating_180d": null,
  "category_diversity_180d": 0, "ticket_count_90d": 3, "negative_ticket_rate_90d": 1.0,
  "avg_resolution_hours_90d": 60.0, "days_since_signup": 700, "sessions_30d": 0,
  "product_views_30d": 0, "cart_adds_30d": 0, "wishlist_adds_30d": 0,
  "abandoned_carts_30d": 0, "email_opens_30d": 0, "campaign_clicks_30d": 0,
  "last_visit_days_ago": 45
}'
```

### Sample response

```json
{
  "churn_probability": 0.9624,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Elevated churn risk: no purchase in 200 days, last app/web visit 45 days ago, low purchase frequency, narrow category breadth, recent negative support sentiment, high return rate."
}
```

`loyalty_tier` and `avg_rating_180d` may be `null` (not enrolled / no ratings). Unknown fields, out-of-range
values, or invalid categories are rejected with HTTP **422**.

## Running the tests

```bash
pytest            # the suite trains model.pkl automatically if it is missing
```

Covers: `/health` → 200; `/predict` valid payload → well-formed response; invalid payload → 422; missing field
→ 422; a high-risk customer scores higher than a low-risk one; `/batch_predict` returns one prediction per input.

## Model & data notes

- The model is the tuned XGBoost pipeline from **Part 3**, retrained here by `train_model.py` from the committed
  `data/rfm_modeling_snapshot.csv` so this repo is fully self-contained and reproducible.
- Features are leakage-safe (only data available on/before the `2025-09-30` snapshot). The decision threshold
  (~0.35) is recall-leaning: missing a churner costs more than a cheap, wasted retention touch.
- The score is **decision-support only**. See `monitoring_plan.md` for the monitoring strategy and the
  responsible-use guidelines (how the retention team should and should not use the output).
