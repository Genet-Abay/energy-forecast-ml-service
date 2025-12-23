from fastapi import FastAPI
from pydantic import BaseModel
import joblib

MODEL_PATH = "models/energy_model.joblib"

app = FastAPI(title="Energy Forecast ML Service")

# Load model at startup
model = joblib.load(MODEL_PATH)


class EnergyInput(BaseModel):
    temperature: float
    hour: int
    day_of_week: int


# @app.get("/health")
# def health_check():
#     return {"status": "ok"}


@app.post("/predict")
def predict(input_data: EnergyInput):
    features = [[
        input_data.temperature,
        input_data.hour,
        input_data.day_of_week
    ]]
    prediction = model.predict(features)[0]
    return {"energy_consumption": round(float(prediction), 2)}
