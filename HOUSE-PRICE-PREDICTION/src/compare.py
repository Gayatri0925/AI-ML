import pandas as pd
import joblib
from .data_loader import load_data

def compare_in_location(location):
    df = load_data("data/house_data.csv")
    model = joblib.load("models/linear_regression.joblib")

    df["predicted_price"] = model.predict(df)
    subset = df[df["location"] == location].copy()

    comparisons = []
    for _, row in subset.iterrows():
        reasons = []
        if row["smart_home"]: reasons.append("Smart Home")
        if row["green_certified"]: reasons.append("Green Certified")
        if row["iot_enabled"]: reasons.append("IoT Enabled")
        if row["tech_score"] > 7: reasons.append("High Tech Score")
        if row["age_of_house"] < 5: reasons.append("New Construction")

        comparisons.append({
            "sqft": row["sqft"],
            "predicted_price": row["predicted_price"],
            "reasons": ", ".join(reasons) if reasons else "Basic Features"
        })

    return pd.DataFrame(comparisons)
