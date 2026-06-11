from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import PredictionRequest, PredictionResponse

app = FastAPI(title="Credit Risk Platform")

@app.on_event("startup")
def load_model():
    app.state.model = None

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Placeholder inference logic
    score = 0.0
    return PredictionResponse(score=score, approved=False)
