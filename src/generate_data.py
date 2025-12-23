import pandas as pd
import numpy as np

def generate_energy_data(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    temperature = np.random.normal(loc=15, scale=10, size=n_samples)
    hour = np.random.randint(0, 24, size=n_samples)
    day_of_week = np.random.randint(0, 7, size=n_samples)

    base_consumption = 50
    temp_effect = (20 - temperature) * 1.5
    hour_effect = np.where((hour >= 8) & (hour <= 18), 20, -5)
    weekday_effect = np.where(day_of_week < 5, 10, -10)

    noise = np.random.normal(0, 5, size=n_samples)

    energy_consumption = (
        base_consumption
        + temp_effect
        + hour_effect
        + weekday_effect
        + noise
    )

    df = pd.DataFrame({
        "temperature": temperature,
        "hour": hour,
        "day_of_week": day_of_week,
        "energy_consumption": energy_consumption
    })

    return df


if __name__ == "__main__":
    df = generate_energy_data()
    df.to_csv("data/energy.csv", index=False)
    print("Dataset saved to data/energy.csv")
