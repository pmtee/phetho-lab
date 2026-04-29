import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os

MODEL_PATH   = "models/titanic_model.pkl"
FEATURE_PATH = "models/features.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model    = joblib.load(MODEL_PATH)
features = joblib.load(FEATURE_PATH)

app = FastAPI(title="Titanic Survival Predictor API", version="1.0.0")

class Passenger(BaseModel):
    pclass:   int   = Field(..., ge=1, le=3)
    sex:      str   = Field(...)
    age:      float = Field(..., ge=0, le=120)
    sibsp:    int   = Field(0, ge=0)
    parch:    int   = Field(0, ge=0)
    fare:     float = Field(..., ge=0)
    embarked: str   = Field("S")

def encode_sex(sex):
    s = sex.strip().lower()
    if s not in ("male", "female"):
        raise HTTPException(status_code=422, detail=f"sex must be male or female")
    return 1 if s == "female" else 0

def encode_embarked(port):
    p = port.strip().upper()
    mapping = {"S": 0, "C": 1, "Q": 2}
    if p not in mapping:
        raise HTTPException(status_code=422, detail=f"embarked must be S C or Q")
    return mapping[p]

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "RandomForestClassifier", "accuracy": "82.7%", "docs": "/docs"}

@app.get("/model-info")
def model_info():
    return {"model_type": type(model).__name__, "n_estimators": model.n_estimators, "features": features, "accuracy": "82.7%"}

@app.post("/predict")
def predict(passenger: Passenger):
    sex_num      = encode_sex(passenger.sex)
    embarked_num = encode_embarked(passenger.embarked)
    data = pd.DataFrame([{
        "Pclass": passenger.pclass, "Sex_num": sex_num,
        "Age": passenger.age, "SibSp": passenger.sibsp,
        "Parch": passenger.parch, "Fare": passenger.fare,
        "Embarked_num": embarked_num
    }])
    prediction  = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {
        "survived": prediction,
        "prediction": "SURVIVED" if prediction == 1 else "DIED",
        "probability": round(probability, 4),
        "confidence": f"{probability * 100:.1f}%"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
