
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os

# ── Load model once at startup ──────────────────
MODEL_PATH   = "models/titanic_model.pkl"
FEATURE_PATH = "models/features.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run the training notebook first."
    )

model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

# ── FastAPI app ──────────────────────────────────
app = FastAPI(
    title="Titanic Survival Predictor API",
    description=(
        "Predicts whether a Titanic passenger would "
        "survive based on their details. "
        "Built by Phetho Tlaka | pmtee.github.io"
    ),
    version="1.0.0"
)

# ── Input schema ─────────────────────────────────
# Pydantic validates every incoming request
# If a field is missing or wrong type → 422 error
class Passenger(BaseModel):
    pclass:   int   = Field(..., ge=1, le=3,
                      description="Ticket class 1, 2 or 3")
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
                      description="Port: S, C or Q")

# ── Encode helpers ───────────────────────────────
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

# ── Routes ───────────────────────────────────────
@app.get("/")
def health_check():
    return {
        "status":  "healthy",
        "model":   "RandomForestClassifier",
        "accuracy":"82.7%",
        "author":  "Phetho Tlaka",
        "docs":    "/docs"
    }

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

@app.post("/predict")
def predict(passenger: Passenger):
    # Encode text fields to numbers
    sex_num      = encode_sex(passenger.sex)
    embarked_num = encode_embarked(passenger.embarked)

    # Build feature DataFrame in correct order
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
    probability = float(
        model.predict_proba(data)[0][1])

    return {
        "survived":    prediction,
        "prediction":  "SURVIVED" if prediction == 1
                       else "DIED",
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

# ── Run ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
