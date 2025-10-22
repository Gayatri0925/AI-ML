import pandas as pd
import numpy as np
import joblib
from .data_loader import load_data
from .preprocess import build_preprocessor

def train_and_predict():
    # Load dataset
    df = load_data("data/house_data.csv")

    # Simulated price (hidden from CSV, but generated for training)
    # Formula: base on sqft, bedrooms, green, tech, IoT, minus age effect
    df["price"] = (df["sqft"] * 50) + (df["bedrooms"] * 5000) + (df["bathrooms"] * 8000) \
                  + (df["smart_home"] * 10000) + (df["green_certified"] * 12000) \
                  + (df["tech_score"] * 1500) + (df["iot_enabled"] * 8000) \
                  - (df["age_of_house"] * 500) + np.random.randint(-5000, 5000, size=len(df))

    X = df.drop("price", axis=1)
    y = df["price"]

    # Build model
    model = build_preprocessor()
    model.fit(X, y)

    # Save model
    joblib.dump(model, "models/linear_regression.joblib")

    # Predictions
    df["predicted_price"] = model.predict(X)
    return df[["location", "sqft", "bedrooms", "bathrooms", 
               "age_of_house", "smart_home", "green_certified", 
               "tech_score", "iot_enabled", "predicted_price"]]
