"""
simulate.py
-----------
Generates simulated future environmental data for waste prediction.
Each row represents one future day for a given city / sector / zone.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

# Import shared utilities
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import FEATURE_COLS, get_city_meta


def _get_season(month: int) -> int:
    """
    Map a month number to a season encoding.
    0=Post-Monsoon, 1=Rainy, 2=Summer, 3=Winter
    """
    if month in (12, 1, 2):
        return 3   # Winter
    elif month in (3, 4, 5):
        return 2   # Summer
    elif month in (6, 7, 8, 9):
        return 1   # Rainy
    else:
        return 0   # Post-Monsoon (Oct, Nov)


def generate_future_data(
    days: int,
    city_name: str,
    city_enc: int,
    sector_enc: int,
    zone_enc: int,
    start_date: date = None,
) -> pd.DataFrame:
    """
    Simulate future daily environmental data for a given city/sector/zone.

    Parameters
    ----------
    days        : Number of future days to simulate (1–14)
    city_name   : City name string (used to look up population/density)
    city_enc    : Encoded city index (0–17)
    sector_enc  : Encoded sector index (0–3)
    zone_enc    : Encoded zone index (0–4)
    start_date  : First date to simulate (defaults to tomorrow)

    Returns
    -------
    pandas DataFrame with FEATURE_COLS columns, ready for model prediction.
    """
    if start_date is None:
        start_date = date.today() + timedelta(days=1)

    # Fixed random seed for reproducibility within a session
    rng = np.random.default_rng(seed=42)

    # City-level static features
    meta               = get_city_meta(city_name)
    population         = meta["population"]
    area_km2           = meta["area_km2"]
    population_density = round(population / area_km2, 2)

    rows = []
    for i in range(days):
        current_date = start_date + timedelta(days=i)

        # ── Simulate weather features with realistic daily variation ──────────
        # Temperature: base 30°C ± 5°C seasonal noise
        temperature = round(30.0 + rng.normal(0, 5), 1)
        temperature = float(np.clip(temperature, 14, 47))

        # Humidity: base 60% ± 20% variation
        humidity = int(np.clip(round(60 + rng.normal(0, 20)), 18, 100))

        # Pollution index: base 100 AQI ± 30 variation
        pollution_index = round(
            float(np.clip(100 + rng.normal(0, 30), 20, 250)), 1
        )

        # ── Time features ─────────────────────────────────────────────────────
        year        = current_date.year
        month       = current_date.month
        day_of_week = current_date.weekday()   # 0=Monday … 6=Sunday
        is_weekend  = 1 if day_of_week >= 5 else 0

        # ── Season derived from month ─────────────────────────────────────────
        season_enc = _get_season(month)

        # ── Weather condition: 0=Clear (default for future simulation) ────────
        weather_condition_enc = 0

        rows.append({
            "date":                   current_date.strftime("%Y-%m-%d"),
            "temperature":            temperature,
            "humidity":               humidity,
            "population":             population,
            "pollution_index":        pollution_index,
            "population_density":     population_density,
            "year":                   year,
            "month":                  month,
            "day_of_week":            day_of_week,
            "is_weekend":             is_weekend,
            "sector_enc":             sector_enc,
            "weather_condition_enc":  weather_condition_enc,
            "location_enc":           city_enc,
            "season_enc":             season_enc,
            "zone_enc":               zone_enc,
        })

    df = pd.DataFrame(rows)
    return df
