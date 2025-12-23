import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from model import create_model


DATA_PATH = "data/energy.csv"
MODEL_PATH = "models/energy_model.joblib"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    X = df[["temperature", "hour", "day_of_week"]]
    y = df["energy_consumption"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = create_model()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Validation MAE: {mae:.2f}")

    # Save model
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
