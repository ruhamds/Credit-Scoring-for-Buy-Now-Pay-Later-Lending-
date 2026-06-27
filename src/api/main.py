"""
main.py
───────
FastAPI credit risk scoring service.

Endpoints:
    GET  /health    → liveness + readiness check
    POST /predict   → risk score + decision + SHAP factors + audit log
    GET  /schema    → expected input fields and types

Model loaded once at startup via lifespan — not per request.
SHAP explainer initialised alongside model at startup.
Every prediction written to PostgreSQL audit log.
"""

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Depends
from sqlalchemy.orm import Session, sessionmaker

from src.api.pydantic_models import (
    PredictRequest, PredictResponse, RiskFactor, HealthResponse
)
from src.api.prediction_log import (
    get_engine, init_db, log_prediction, PredictionRecord
)
from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
MODEL_VERSION    = "1.0.0"
DECISION_BANDS   = {
    "approve": (0.00, 0.40),   # P(default) < 0.40 → approve
    "refer"  : (0.40, 0.65),   # 0.40–0.65 → refer to human review
    "reject" : (0.65, 1.01),   # > 0.65    → reject
}
FEATURE_COLS = [
    "Amount_count", "Amount_sum", "Amount_mean", "Amount_std",
    "Amount_min", "Amount_max", "tx_hour_mean", "tx_day_mean",
    "tx_month_mean", "tx_dayofweek_mean", "tx_is_weekend_mean",
    "ProviderId_mode", "ChannelId_mode", "ProductCategory_mode",
]


# ── Lifespan — load model + explainer + DB once at startup ─────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):

    # ── Validate all required artifacts exist before loading anything
    required_artifacts = [
        "model.joblib",
        "selected_features.csv",
        "model_threshold.joblib",
    ]
    for name in required_artifacts:
        path = ARTIFACTS_DIR / name
        if not path.exists():
            raise RuntimeError(
                f"Missing required artifact: {name}. "
                f"Run src/train.py before starting the API."
            )

    # ── Model
    app.state.model = joblib.load(ARTIFACTS_DIR / "model.joblib")
    logger.info("Model loaded")

    # ── Feature columns — loaded from training artifact, not hardcoded
    app.state.feature_cols = (
        pd.read_csv(ARTIFACTS_DIR / "selected_features.csv")
        .iloc[:, 0]
        .tolist()
    )
    logger.info(f"Feature cols loaded: {app.state.feature_cols}")

    # ── Threshold
    app.state.threshold = joblib.load(ARTIFACTS_DIR / "model_threshold.joblib")
    logger.info(f"Decision threshold: {app.state.threshold}")

    # ── SHAP explainer
    app.state.explainer = None
    try:
        base   = getattr(app.state.model, "estimator", app.state.model)
        sample = pd.DataFrame(
            np.zeros((1, len(app.state.feature_cols))),
            columns=app.state.feature_cols
        )
        try:
            app.state.explainer = shap.LinearExplainer(base, sample)
        except Exception:
            app.state.explainer = shap.KernelExplainer(
                app.state.model.predict_proba, sample
            )
        logger.info("SHAP explainer initialised")
    except Exception as e:
        logger.warning(f"SHAP explainer failed: {e}")

    # ── Database
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://creditrisk:creditrisk@localhost:5432/creditrisk"
    )
    try:
        engine = get_engine(db_url)
        init_db(engine)
        app.state.db_engine   = engine
        app.state.SessionLocal = sessionmaker(bind=engine)
        app.state.db_ok       = True
        logger.info("Database connected")
    except Exception as e:
        logger.warning(f"Database unavailable: {e} — predictions will not be logged")
        app.state.db_engine   = None
        app.state.SessionLocal = None
        app.state.db_ok       = False

    yield

    if app.state.db_engine:
        app.state.db_engine.dispose()
    app.state.model = None


# ── App ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Credit Risk Scoring API — Bati Bank",
    description = "Probability of default scoring for BNPL credit decisions",
    version     = MODEL_VERSION,
    lifespan    = lifespan,
)


# ── DB session dependency ──────────────────────────────────────────────────

def get_db():
    """Yields a DB session if available, else None."""
    session_local = getattr(app.state, "SessionLocal", None)
    if session_local is None:
        yield None
        return
    session = session_local()
    try:
        yield session
    finally:
        session.close()


# ── Helper: derive decision from score ────────────────────────────────────

def score_to_decision(score: float) -> str:
    for decision, (low, high) in DECISION_BANDS.items():
        if low <= score < high:
            return decision
    return "reject"


# ── Helper: SHAP top factors ──────────────────────────────────────────────

def get_top_factors(
    explainer   ,
    input_df    : pd.DataFrame,
    feature_cols: list,
    n           : int = 3,
) -> list[dict]:
    if explainer is None:
        return []
    try:
        shap_values = explainer.shap_values(input_df)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        if hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
            shap_values = shap_values[0]

        pairs = sorted(
            zip(feature_cols, shap_values.tolist()),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:n]

        return [
            {
                "feature"  : name,
                "impact"   : round(val, 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            }
            for name, val in pairs
        ]
    except Exception as e:
        logger.warning(f"SHAP factor extraction failed: {e}")
        return []


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Liveness + readiness check.
    Returns model load status and DB connectivity.
    Used by Docker HEALTHCHECK and CI/CD pipeline.
    """
    return HealthResponse(
        status        = "ok" if getattr(app.state, "model", None) else "degraded",
        model_loaded  = getattr(app.state, "model", None) is not None,
        model_version = MODEL_VERSION,
        db_connected  = getattr(app.state, "db_ok", False),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    db     : Optional[Session] = Depends(get_db),
):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_cols = app.state.feature_cols
    features     = request.model_dump()
    input_df     = pd.DataFrame([features])[feature_cols]

    risk_score  = float(model.predict_proba(input_df)[0][1])
    decision    = score_to_decision(risk_score)
    top_factors = get_top_factors(
        app.state.explainer, input_df, feature_cols
    )

    prediction_id = str(uuid.uuid4())
    if db is not None:
        try:
            log_prediction(
                session       = db,
                prediction_id = prediction_id,
                features      = features,
                risk_score    = risk_score,
                decision      = decision,
                top_factors   = top_factors,
                model_version = MODEL_VERSION,
                threshold     = app.state.threshold,
            )
        except Exception as e:
            logger.warning(f"Prediction logging failed: {e}")

    return PredictResponse(
        prediction_id  = prediction_id,
        risk_score     = round(risk_score, 4),
        decision       = decision,
        top_factors    = [RiskFactor(**f) for f in top_factors],
        model_version  = MODEL_VERSION,
        threshold_used = app.state.threshold,
    )

@app.get("/schema")
def schema():
    """Returns expected input fields and types for /predict."""
    return {
        "endpoint"      : "/predict",
        "method"        : "POST",
        "input_fields"  : FEATURE_COLS,
        "model_version" : MODEL_VERSION,
        "decision_bands": DECISION_BANDS,
    }