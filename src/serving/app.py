from fastapi import FastAPI, HTTPException
from .schemas import PredictRequest, PredictResponse
from .model_loader import load_artifact
from .predictor import build_predictor


app = FastAPI(title="Avazu CTR Serving API")

artifact = load_artifact()
predict_one = build_predictor(artifact)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        proba = predict_one(req.features)
        pred = 1 if proba >= 0.5 else 0
        return PredictResponse(click_probability=proba, click_prediction=pred)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
