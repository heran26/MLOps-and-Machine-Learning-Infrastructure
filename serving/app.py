"""
FastAPI inference server for the current production model.

Run locally:   uvicorn app:app --host 0.0.0.0 --port 8000
Run on server:  same command, kept alive via systemd (see README.md)
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get("MODEL_PATH", "models/production_model.pkl")

app = FastAPI(title="Income Prediction API")
model = joblib.load(MODEL_PATH)


class PredictionRequest(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    df = pd.DataFrame([request.dict()])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    return {
        "prediction": int(prediction),
        "probability_over_50k": float(probability),
    }
