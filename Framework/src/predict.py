"""
predict.py
----------
Prediction engine for the Smart City Waste Prediction Framework.
Converts user inputs → simulated data → model predictions.
"""

import sys, os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils    import load_model, FEATURE_COLS, CITY_MAP, SECTOR_MAP, ZONE_MAP
from simulate import generate_future_data


def predict_future(
    city: str,
    sector: str,
    zone: str,
    days: int = 7,
) -> pd.DataFrame:
    """
    Full prediction pipeline: inputs → simulation → model → results.

    Parameters
    ----------
    city   : City name string  (e.g. "Delhi")
    sector : Sector name string (e.g. "Industrial")
    zone   : Zone name string   (e.g. "Central")
    days   : Number of future days to predict (1–14)

    Returns
    -------
    pandas DataFrame with columns:
        date, temperature, humidity, pollution_index,
        population_density, season_enc, predicted_waste
    """
    # ── Step 1: Encode string inputs to integers ──────────────────────────────
    city_enc   = CITY_MAP.get(city, 0)
    sector_enc = SECTOR_MAP.get(sector, 0)
    zone_enc   = ZONE_MAP.get(zone, 0)

    # ── Step 2: Simulate future environmental data ────────────────────────────
    df = generate_future_data(
        days=days,
        city_name=city,
        city_enc=city_enc,
        sector_enc=sector_enc,
        zone_enc=zone_enc,
    )

    # ── Step 3: Load trained model ────────────────────────────────────────────
    model = load_model()

    # ── Step 4: Predict waste for each future day ─────────────────────────────
    X = df[FEATURE_COLS]
    predictions = model.predict(X)

    # Clip to realistic range (50–1200 tons/day)
    df["predicted_waste"] = predictions.clip(50, 1200).round(2)

    # ── Step 5: Return clean results DataFrame ────────────────────────────────
    result_cols = [
        "date", "temperature", "humidity",
        "pollution_index", "population_density",
        "season_enc", "predicted_waste",
    ]
    return df[result_cols].reset_index(drop=True)
