from src.train import train_and_predict
from src.compare import compare_in_location
import pandas as pd

def main():
    print("🏡 Training model and predicting prices...")
    df = train_and_predict()
    print("\n✅ Predictions complete!\n")
    print(df.head())  # Show first few predictions

    # Example: Compare houses in Downtown
    print("\n📊 Comparing houses in Downtown:\n")
    comparison = compare_in_location("Downtown")
    print(comparison)

if __name__ == "__main__":
    main()
