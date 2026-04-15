"""
utils.py
--------
Shared utilities for the Smart City Waste Prediction Framework.
Provides model loading, encoding maps, and city population/density data.
"""

import os
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "waste_model.pkl")

# ── Encoding maps (must match training-time label encoding) ───────────────────
SECTOR_MAP = {
    "Commercial":  0,
    "Healthcare":  1,
    "Industrial":  2,
    "Residential": 3,
}

ZONE_MAP = {
    "Central": 0,
    "East":    1,
    "North":   2,
    "South":   3,
    "West":    4,
}

CITY_MAP = {
    "Ahmedabad":     0,
    "Bangalore":     1,
    "Bhopal":        2,
    "Chennai":       3,
    "Coimbatore":    4,
    "Delhi":         5,
    "Hyderabad":     6,
    "Indore":        7,
    "Jaipur":        8,
    "Kochi":         9,
    "Kolkata":       10,
    "Lucknow":       11,
    "Mumbai":        12,
    "Nagpur":        13,
    "Patna":         14,
    "Pune":          15,
    "Surat":         16,
    "Visakhapatnam": 17,
}

# Reverse maps for display
SECTOR_DECODE  = {v: k for k, v in SECTOR_MAP.items()}
ZONE_DECODE    = {v: k for k, v in ZONE_MAP.items()}
CITY_DECODE    = {v: k for k, v in CITY_MAP.items()}

# ── City metadata (population and area for density calculation) ───────────────
CITY_META = {
    "Ahmedabad":     {"population": 8_450_000,  "area_km2": 505},
    "Bangalore":     {"population": 13_190_000, "area_km2": 741},
    "Bhopal":        {"population": 2_400_000,  "area_km2": 463},
    "Chennai":       {"population": 9_110_000,  "area_km2": 426},
    "Coimbatore":    {"population": 2_150_000,  "area_km2": 246},
    "Delhi":         {"population": 32_940_000, "area_km2": 1484},
    "Hyderabad":     {"population": 10_350_000, "area_km2": 650},
    "Indore":        {"population": 3_500_000,  "area_km2": 530},
    "Jaipur":        {"population": 3_950_000,  "area_km2": 467},
    "Kochi":         {"population": 2_120_000,  "area_km2": 94},
    "Kolkata":       {"population": 14_850_000, "area_km2": 185},
    "Lucknow":       {"population": 3_680_000,  "area_km2": 349},
    "Mumbai":        {"population": 20_670_000, "area_km2": 603},
    "Nagpur":        {"population": 2_900_000,  "area_km2": 227},
    "Patna":         {"population": 2_350_000,  "area_km2": 136},
    "Pune":          {"population": 7_280_000,  "area_km2": 331},
    "Surat":         {"population": 7_180_000,  "area_km2": 395},
    "Visakhapatnam": {"population": 2_100_000,  "area_km2": 682},
}

# Feature columns expected by the model (must match training order)
FEATURE_COLS = [
    "temperature", "humidity", "population", "pollution_index",
    "population_density", "year", "month", "day_of_week", "is_weekend",
    "sector_enc", "weather_condition_enc", "location_enc",
    "season_enc", "zone_enc",
]


def load_model():
    """
    Load the trained XGBoost model from disk.

    Returns
    -------
    Trained XGBRegressor model object.

    Raises
    ------
    FileNotFoundError if the model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Please ensure 'waste_model.pkl' is inside the 'models/' folder."
        )
    model = joblib.load(MODEL_PATH)
    return model


def get_city_meta(city_name: str) -> dict:
    """Return population and area_km2 for a given city name."""
    return CITY_META.get(city_name, {"population": 5_000_000, "area_km2": 400})
