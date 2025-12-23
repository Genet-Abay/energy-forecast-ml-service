import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from src.model import create_model

def test_training_runs():
    # small fake dataset
    X = pd.DataFrame({
        "temperature": [10, 15],
        "hour": [12, 18],
        "day_of_week": [0, 1]
    })
    y = [50, 60]

    model = create_model()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == 2
