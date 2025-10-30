# train.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from joblib import dump
import os

def generate_synthetic(n=5000, random_state=42):
    rng = np.random.RandomState(random_state)
    # Features:
    # screen_time (minutes/day), typing_speed (chars per minute),
    # app_switches_per_hour, orientation_changes_per_hour, notifications_per_hour
    screen_time = rng.normal(240, 90, n).clip(10, 1440)
    typing_speed = rng.normal(200, 70, n).clip(10, 800)
    app_switches = rng.poisson(8, n).astype(float)
    orientation = rng.poisson(15, n).astype(float)
    notifications = rng.poisson(12, n).astype(float)

    # Label rules (simple heuristics + noise)
    score = (
        -0.01 * screen_time 
        + 0.003 * typing_speed 
        - 0.2 * (app_switches > 12) 
        - 0.1 * (orientation > 25) 
        - 0.1 * (notifications > 18)
        + rng.normal(0, 0.5, n)
    )
    # map score to labels
    labels = np.where(score > 0.8, "Happy",
             np.where(score > 0.0, "Neutral",
             np.where(score > -0.8, "Tired", "Stressed")))

    df = pd.DataFrame({
        "screen_time": screen_time,
        "typing_speed": typing_speed,
        "app_switches": app_switches,
        "orientation_changes": orientation,
        "notifications": notifications,
        "label": labels
    })
    return df

def train_and_save(out_dir="models"):
    os.makedirs(out_dir, exist_ok=True)
    df = generate_synthetic(n=7000)
    X = df.drop(columns=["label"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    acc = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")

    model_path = os.path.join(out_dir, "smartmood_model.joblib")
    dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

if __name__ == "__main__":
    train_and_save()
