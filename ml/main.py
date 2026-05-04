# ================================================
# Titanic Survival Predictor — FastAPI
# Phetho Tlaka | Week 2 Day 2 | pmtee.github.io
# ================================================

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os

# Load model once when server starts
# Not on every request — that would be slow
MODEL_PATH   = "models/titanic_model.pkl"
FEATURE_PATH = "models/features.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run week2_day2_model_api.ipynb first."
    )

model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

# Create the FastAPI app
app = FastAPI(
    title="Titanic Survival Predictor API",
    description="Predicts Titanic survival. Built by Phetho Tlaka | pmtee.github.io",
    version="1.0.0"
)

# Allow browser requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Input validation — Pydantic checks every request
# Wrong type or out of range → 422 error automatically
class Passenger(BaseModel):
    pclass:   int   = Field(..., ge=1, le=3,
                      description="Ticket class: 1, 2 or 3")
    sex:      str   = Field(...,
                      description="male or female")
    age:      float = Field(..., ge=0, le=120,
                      description="Age in years")
    sibsp:    int   = Field(0, ge=0,
                      description="Siblings/spouses aboard")
    parch:    int   = Field(0, ge=0,
                      description="Parents/children aboard")
    fare:     float = Field(..., ge=0,
                      description="Ticket price in pounds")
    embarked: str   = Field("S",
                      description="Port: S=Southampton C=Cherbourg Q=Queenstown")

# Convert text to numbers — ML only understands numbers
def encode_sex(sex: str) -> int:
    s = sex.strip().lower()
    if s not in ("male", "female"):
        raise HTTPException(
            status_code=422,
            detail=f"sex must be male or female, got: {sex}"
        )
    return 1 if s == "female" else 0

def encode_embarked(port: str) -> int:
    p = port.strip().upper()
    mapping = {"S": 0, "C": 1, "Q": 2}
    if p not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"embarked must be S, C or Q, got: {port}"
        )
    return mapping[p]

# Health check — is the API alive?
@app.get("/")
def health_check():
    return {
        "status":   "healthy",
        "model":    "RandomForestClassifier",
        "accuracy": "82.7%",
        "author":   "Phetho Tlaka",
        "portfolio":"pmtee.github.io",
        "docs":     "/docs"
    }

# Model info — what is running?
@app.get("/model-info")
def model_info():
    return {
        "model_type":    type(model).__name__,
        "n_estimators":  model.n_estimators,
        "features":      features,
        "n_features":    len(features),
        "accuracy":      "82.7%",
        "trained_on":    "891 Titanic passengers",
        "test_set_size": "179 passengers"
    }

# Prediction endpoint — main endpoint
@app.post("/predict")
def predict(passenger: Passenger):
    # Convert text fields to numbers
    sex_num      = encode_sex(passenger.sex)
    embarked_num = encode_embarked(passenger.embarked)

    # Build DataFrame in exact same order as training
    data = pd.DataFrame([{
        "Pclass":       passenger.pclass,
        "Sex_num":      sex_num,
        "Age":          passenger.age,
        "SibSp":        passenger.sibsp,
        "Parch":        passenger.parch,
        "Fare":         passenger.fare,
        "Embarked_num": embarked_num
    }])

    # Make prediction
    prediction  = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "survived":    prediction,
        "prediction":  "SURVIVED" if prediction == 1 else "DIED",
        "probability": round(probability, 4),
        "confidence":  f"{probability * 100:.1f}%",
        "input": {
            "pclass":   passenger.pclass,
            "sex":      passenger.sex,
            "age":      passenger.age,
            "fare":     passenger.fare,
            "embarked": passenger.embarked
        }
    }

# Start server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)