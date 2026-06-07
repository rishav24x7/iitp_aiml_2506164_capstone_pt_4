"""Pydantic request/response schemas for the churn scoring API.

The CustomerFeatures model mirrors the leakage-safe feature set used to train the model in Part 3.
Field constraints provide input validation (non-negative counts, bounded rates) so malformed payloads
are rejected with a 422 before reaching the model.
"""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

CityTier = Literal["Tier 1", "Tier 2", "Tier 3"]
AgeGroup = Literal["18-24", "25-34", "35-44", "45+"]
AcqChannel = Literal["Google Search", "Instagram", "Influencer", "Referral", "Marketplace", "Organic"]
Category = Literal["Skin Care", "Hair Care", "Makeup", "Fragrance", "Wellness", "Baby Care"]
Consent = Literal["Yes", "No"]
LoyaltyTier = Literal["Silver", "Gold", "Platinum"]


class CustomerFeatures(BaseModel):
    """One customer's snapshot features. Matches rfm_modeling_snapshot.csv (minus IDs/target)."""
    model_config = {"extra": "forbid"}  # reject unknown fields -> 422

    # Profile (categorical)
    city_tier: CityTier
    age_group: AgeGroup
    acquisition_channel: AcqChannel
    preferred_category: Category
    marketing_consent: Consent
    loyalty_tier: Optional[LoyaltyTier] = Field(
        default=None, description="Null/None means the customer is not enrolled in the loyalty programme.")

    # RFM & behavioural (numeric, constrained)
    recency_days: int = Field(ge=0, le=10000)
    frequency_180d: int = Field(ge=0, le=1000)
    monetary_180d: float = Field(ge=0)
    return_rate_180d: float = Field(ge=0.0, le=1.0)
    avg_discount_pct_180d: float = Field(ge=0.0, le=1.0)
    avg_rating_180d: Optional[float] = Field(default=None, ge=1.0, le=5.0,
                                             description="Null if the customer left no ratings.")
    category_diversity_180d: int = Field(ge=0, le=20)
    ticket_count_90d: int = Field(ge=0, le=1000)
    negative_ticket_rate_90d: float = Field(ge=0.0, le=1.0)
    avg_resolution_hours_90d: float = Field(ge=0.0)
    days_since_signup: int = Field(ge=0, le=10000)
    sessions_30d: int = Field(ge=0, le=10000)
    product_views_30d: int = Field(ge=0, le=100000)
    cart_adds_30d: int = Field(ge=0, le=100000)
    wishlist_adds_30d: int = Field(ge=0, le=100000)
    abandoned_carts_30d: int = Field(ge=0, le=100000)
    email_opens_30d: int = Field(ge=0, le=100000)
    campaign_clicks_30d: int = Field(ge=0, le=100000)
    last_visit_days_ago: int = Field(ge=0, le=10000)

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "example": {
                "city_tier": "Tier 2", "age_group": "25-34", "acquisition_channel": "Instagram",
                "preferred_category": "Skin Care", "marketing_consent": "Yes", "loyalty_tier": "Silver",
                "recency_days": 95, "frequency_180d": 1, "monetary_180d": 640.0,
                "return_rate_180d": 0.0, "avg_discount_pct_180d": 0.22, "avg_rating_180d": 3.5,
                "category_diversity_180d": 1, "ticket_count_90d": 2, "negative_ticket_rate_90d": 1.0,
                "avg_resolution_hours_90d": 40.0, "days_since_signup": 300, "sessions_30d": 1,
                "product_views_30d": 3, "cart_adds_30d": 0, "wishlist_adds_30d": 0,
                "abandoned_carts_30d": 1, "email_opens_30d": 1, "campaign_clicks_30d": 0,
                "last_visit_days_ago": 28,
            }
        },
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0.0, le=1.0)
    predicted_class: int = Field(ge=0, le=1)
    risk_level: Literal["low", "medium", "high"]
    risk_explanation: str


class BatchPredictRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
