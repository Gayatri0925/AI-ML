from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def build_preprocessor():
    categorical = ["location"]
    numeric = ["sqft", "bedrooms", "bathrooms", "age_of_house", "smart_home", 
               "green_certified", "tech_score", "iot_enabled"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric)
        ]
    )
    model = Pipeline(steps=[("preprocessor", preprocessor),
                            ("regressor", LinearRegression())])
    return model
