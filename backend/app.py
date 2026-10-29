"""
Imari Health — FastAPI Backend

Exposes the financial health scoring, SHAP explanation, and recommendation
engine as REST endpoints for the frontend.

Run: uvicorn app:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import shap

from scoring import (
    FEATURES, SEGMENT_ORDER, build_explainer, get_feature_names, predict_one
)
from recommendations import get_recommendation

MODEL_DIR = "models"
PIPELINE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")

app = FastAPI(
    title="Imari Health API",
    description="Financial Health Navigator for Rwanda — NISR 2026 Big Data Hackathon",
    version="1.0.0",
)

# CORS for the frontend (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL AT STARTUP ---
if not os.path.exists(PIPELINE_PATH):
    raise RuntimeError(
        f"Model not found at {PIPELINE_PATH}. Run `python scoring.py` first."
    )

pipeline = joblib.load(PIPELINE_PATH)
explainer = build_explainer(pipeline)
feature_names = get_feature_names(pipeline)


# --- REQUEST SCHEMA ---
class UserAnswers(BaseModel):
    gender: str = Field(..., examples=["female"])
    area: str = Field(..., examples=["rural"])
    age_group: str = Field(..., examples=["25-34"])
    education: str = Field(..., examples=["primary_or_below"])
    income_source: str = Field(..., examples=["independent_farmer"])
    has_mobile_money: int = Field(..., ge=0, le=1)
    has_bank_account: int = Field(..., ge=0, le=1)
    has_savings_account: int = Field(..., ge=0, le=1)
    has_insurance: int = Field(..., ge=0, le=1)
    has_pension: int = Field(..., ge=0, le=1)
    has_goal: int = Field(..., ge=0, le=1)
    goal_action: str = Field(..., examples=["nothing"])
    runs_out: str = Field(..., examples=["often"])
    can_raise: str = Field(..., examples=["no"])
    controls_finances: int = Field(..., ge=0, le=1)


# --- ENDPOINTS ---
@app.get("/")
def root():
    return {
        "name": "Imari Health API",
        "status": "running",
        "docs": "/docs",
        "endpoints": ["/predict", "/health"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "features": len(FEATURES)}


@app.post("/predict")
def predict(answers: UserAnswers):
    try:
        result = predict_one(answers.model_dump(), pipeline, explainer, feature_names)
        recommendation = get_recommendation(answers.model_dump(), result)
        return {
            "prediction": result,
            "recommendation": recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)