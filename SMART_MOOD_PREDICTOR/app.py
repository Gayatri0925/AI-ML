# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import os

MODEL_PATH = os.environ.get("SMARTMOOD_MODEL", "models/smartmood_model.joblib")

app = FastAPI(title="SmartMood API")
model = load(MODEL_PATH)

class Features(BaseModel):
    screen_time: float
    typing_speed: float
    app_switches: float
    orientation_changes: float
    notifications: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(feat: Features):
    X = [[
        feat.screen_time,
        feat.typing_speed,
        feat.app_switches,
        feat.orientation_changes,
        feat.notifications
    ]]
    proba = model.predict_proba(X)[0]
    classes = model.classes_.tolist()
    pred = model.predict(X)[0]
    probs = dict(zip(classes, [float(round(p, 4)) for p in proba]))
    suggestions = {
        "Happy": "Keep going 👍 — maybe share this good mood with a friend!",
        "Neutral": "You're steady. A short walk or stretch could boost energy.",
        "Tired": "Consider a 20-minute power nap or a short break.",
        "Stressed": "Try breathing exercises or step away for 10 minutes."
    }
    return {"prediction": pred, "probabilities": probs, "suggestion": suggestions.get(pred, "")}
