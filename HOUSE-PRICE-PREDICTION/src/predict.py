import pandas as pd
import numpy as np
from utils import load_model
from data_loader import load_data

def compare_in_location(location_name, data_path="data/housing.csv", model_path="models/linear_regression.joblib"):
    df = load_data(data_path)
    model = load_model(model_path)

    num_cols = ["area_sqft", "bedrooms", "bathrooms"]
    cat_cols = ["location"]
    features = num_cols + cat_cols

    # Filter location
    subset = df[df["location"] == location_name].copy()
    if subset.empty:
        print(f"No data found for location: {location_name}")
        return

    # Predict
    preds_log = model.predict(subset[features])
    preds = np.expm1(preds_log)

    subset["predicted_price"] = preds
    subset["residual"] = subset["price"] - subset["predicted_price"]
    subset["price_per_sqft"] = subset["price"] / subset["area_sqft"]

    # Sort by predicted price
    result = subset.sort_values(by="predicted_price", ascending=False)
    print(result[["location", "price", "predicted_price", "residual", "price_per_sqft"]])

if __name__ == "__main__":
    compare_in_location("Downtown")  # Change location as needed
