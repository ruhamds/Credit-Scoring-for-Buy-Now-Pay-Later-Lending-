from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib, logging
from src.config import ARTIFACTS_DIR

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ARTIFACTS_DIR / "model.joblib"
    app.state.model = joblib.load(model_path) if model_path.exists() else None
    if app.state.model is None:
        logger.warning("No model found — /predict will return 503")
    yield
    app.state.model = None

app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    loaded = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model": "loaded" if loaded else "not loaded"}

@app.post("/predict")
def predict(request: dict):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    # full implementation comes in Task 6 with Pydantic models
    return {"detail": "not implemented yet"}