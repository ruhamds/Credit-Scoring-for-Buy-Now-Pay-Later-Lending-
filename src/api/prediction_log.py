"""
prediction_log.py
─────────────────
SQLAlchemy model and utilities for prediction audit logging.

Every /predict call is logged to PostgreSQL with:
- prediction_id  (UUID — returned to caller for retrieval)
- input hash     (SHA256 of input features — detects duplicate requests)
- risk score     (calibrated probability)
- decision       (approve/refer/reject)
- model version  (which registered model produced this)
- timestamp      (UTC)

This audit trail satisfies Basel II's requirement that credit decisions
be traceable and reproducible.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, String, Float,
    DateTime, Text, Index
)
from sqlalchemy.orm import declarative_base, Session

logger = logging.getLogger(__name__)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"

    prediction_id = Column(String(36), primary_key=True)
    input_hash = Column(String(64), nullable=False, index=True)
    risk_score = Column(Float, nullable=False)
    decision = Column(String(10), nullable=False)
    top_factors = Column(Text, nullable=False)   # JSON string
    model_version = Column(String(50), nullable=False)
    threshold_used = Column(Float, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("ix_predictions_created_at", "created_at"),
    )


def get_engine(db_url: str):
    return create_engine(
        db_url,
        pool_pre_ping=True,   # detect stale connections
        pool_size=5,
        max_overflow=10,
    )


def init_db(engine) -> None:
    """Creates tables if they don't exist. Idempotent."""
    Base.metadata.create_all(engine)
    logger.info("Prediction audit table initialised")


def hash_input(features: dict) -> str:
    """SHA256 hash of input features — detects duplicate requests."""
    canonical = json.dumps(features, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def log_prediction(
    session: Session,
    prediction_id: str,
    features: dict,
    risk_score: float,
    decision: str,
    top_factors: list,
    model_version: str,
    threshold: float,
) -> None:
    """Writes one prediction record to the audit log."""
    record = PredictionRecord(
        prediction_id=prediction_id,
        input_hash=hash_input(features),
        risk_score=risk_score,
        decision=decision,
        top_factors=json.dumps(top_factors),
        model_version=model_version,
        threshold_used=threshold,
    )
    session.add(record)
    session.commit()
    logger.debug(f"Logged prediction {prediction_id}")
