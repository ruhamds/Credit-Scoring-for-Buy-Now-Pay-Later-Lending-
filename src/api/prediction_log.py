from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(Integer, nullable=False)
    score = Column(Float, nullable=False)
    approved = Column(Boolean, nullable=False)
    request_data = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
