"""
pydantic_models.py
──────────────────
Request and response schemas for the credit risk scoring API.

Field validators enforce business-level constraints — not just types.
A request that passes Pydantic validation is guaranteed to be
interpretable by the model. Invalid inputs return HTTP 422 with
a clear error message, never HTTP 500.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictRequest(BaseModel):
    """
    Customer-level feature vector for risk scoring.
    All fields map directly to selected_features.csv — the contract
    between the feature pipeline and the serving layer.

    Field ranges derived from training data distribution.
    Requests outside these ranges are rejected with HTTP 422.
    """

    # Transaction frequency and volume aggregates
    Amount_count: float = Field(..., ge=1, description="Total transaction count (min 1)")
    Amount_sum: float = Field(..., description="Sum of signed-log-transformed amounts")
    Amount_mean: float = Field(..., description="Mean of signed-log-transformed amounts")
    Amount_std: float = Field(...,
                              ge=0,
                              description="Std dev of amounts (0 for single-tx customers)")
    Amount_min: float = Field(..., description="Min signed-log amount")
    Amount_max: float = Field(..., description="Max signed-log amount")

    # Temporal behaviour aggregates
    tx_hour_mean: float = Field(..., ge=0, le=23, description="Mean transaction hour (0–23)")
    tx_day_mean: float = Field(..., ge=1, le=31, description="Mean transaction day of month")
    tx_month_mean: float = Field(..., ge=1, le=12, description="Mean transaction month")
    tx_dayofweek_mean: float = Field(..., ge=0, le=6, description="Mean day of week (0=Mon, 6=Sun)")
    tx_is_weekend_mean: float = Field(..., ge=0, le=1,
                                      description="Fraction of weekend transactions")

    # Categorical modes (label-encoded)
    ProviderId_mode: float = Field(..., ge=0, description="Most frequent provider (encoded)")
    ChannelId_mode: float = Field(..., ge=0, description="Most frequent channel (encoded)")
    ProductCategory_mode: float = Field(...,
                                        ge=0,
                                        description="Most frequent product category (encoded)")

    @field_validator("Amount_std")
    @classmethod
    def std_non_negative(cls, v):
        if v < 0:
            raise ValueError("Amount_std cannot be negative")
        return v

    @field_validator("tx_is_weekend_mean")
    @classmethod
    def weekend_fraction_valid(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("tx_is_weekend_mean must be between 0 and 1")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Amount_count": 12.0,
                "Amount_sum": 87.3,
                "Amount_mean": 7.3,
                "Amount_std": 2.1,
                "Amount_min": 3.4,
                "Amount_max": 11.2,
                "tx_hour_mean": 14.5,
                "tx_day_mean": 15.2,
                "tx_month_mean": 11.8,
                "tx_dayofweek_mean": 2.3,
                "tx_is_weekend_mean": 0.17,
                "ProviderId_mode": 3.0,
                "ChannelId_mode": 2.0,
                "ProductCategory_mode": 1.0,
            }
        }
    }


class RiskFactor(BaseModel):
    """Single SHAP-derived risk factor for explainability."""
    feature: str
    impact: float = Field(..., description="SHAP value — magnitude indicates importance")
    direction: Literal["increases_risk", "decreases_risk"]


class PredictResponse(BaseModel):
    """
    Full prediction response with audit trail and explainability.

    prediction_id: used to retrieve this prediction from the audit log
    decision:      'approve' | 'refer' | 'reject' based on threshold bands
    top_factors:   top 3 SHAP features driving this specific prediction
    """
    prediction_id: str
    risk_score: float = Field(..., ge=0, le=1)
    decision: Literal["approve", "refer", "reject"]
    top_factors: list[RiskFactor]
    model_version: str
    threshold_used: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    db_connected: bool
