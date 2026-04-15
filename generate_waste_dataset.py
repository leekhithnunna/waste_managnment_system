"""
Smart City Waste Prediction Dataset Generator
----------------------------------------------
- Fetches real weather data from Open-Meteo API (free, no API key needed)
  as a reliable public API for temperature, humidity, and weather conditions.
- Expands to 50,000–60,000 rows using synthetic simulation across
  multiple Indian cities, dates (2 years), and sectors.
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time

# ── 1. City configuration ────────────────────────────────────────────────────
CITIES = {
    "Bangalore":  {"lat": 12.97, "lon": 77.59, "population": 13_190_000},
    "Hyderabad":  {"lat": 17.38, "lon": 78.47, "population": 10_350_000},
    "Chennai":    {"lat": 13.08, "lon": 80.27, "population":  9_110_000},
    "Mumbai":     {"lat": 19.07, "lon": 72.87, "population": 20_670_000},
    "Delhi":      {"lat": 28.61, "lon": 77.20, "population": 32_940_000},
    "Kolkata":    {"lat": 22.57, "lon": 88.36, "population": 14_850_000},
    "Pune":       {"lat": 18.52, "lon": 73.85, "population":  7_280_000},
    "Ahmedabad":  {"lat": 23.02, "lon": 72.57, "population":  8_450_000},
    "Jaipur":     {"lat": 26.91, "lon": 75.79, "population":  3_950_000},
    "Surat":      {"lat": 21.17, "lon": 72.83, "population":  7_180_000},
    "Lucknow":    {"lat": 26.85, "lon": 80.95, "population":  3_680_000},
    "Nagpur":     {"lat": 21.14, "lon": 79.08, "population":  2_900_000},
    "Indore":     {"lat": 22.72, "lon": 75.86, "population":  3_500_000},
    "Bhopal":     {"lat": 23.26, "lon": 77.41, "population":  2_400_000},
    "Visakhapatnam": {"lat": 17.69, "lon": 83.22, "population": 2_100_000},
    "Patna":      {"lat": 25.59, "lon": 85.13, "population":  2_350_000},
    "Coimbatore": {"lat": 11.00, "lon": 76.96, "population":  2_150_000},
    "Kochi":      {"lat":  9.93, "lon": 76.26, "population":  2_120_000},
}

# Number of zones per city (North, South, East, West, Central) — multiplies rows by 5
ZONES = ["North", "South", "East", "West", "Central"]

SECTORS = ["Residential", "Commercial", "Industrial", "Healthcare"]

# Sector weights control waste contribution (Industrial > Commercial > Residential > Healthcare)
SECTOR_WEIGHTS = {
    "Residential": 120,
    "Commercial":  200,
    "Industrial":  350,
    "Healthcare":   80,
}

# WMO weather-code → human-readable condition mapping (Open-Meteo uses WMO codes)
WMO_MAP = {
    0: "Clear", 1: "Clear", 2: "Partly Cloudy", 3: "Clouds",
    45: "Fog", 48: "Fog",
    51: "Drizzle", 53: "Drizzle", 55: "Drizzle",
    61: "Rain", 63: "Rain", 65: "Heavy Rain",
    71: "Snow", 73: "Snow", 75: "Heavy Snow",
    80: "Rain", 81: "Rain", 82: "Heavy Rain",
    95: "Thunderstorm", 96: "Thunderstorm", 99: "Thunderstorm",
}

# ── 2. Fetch real weather data from Open-Meteo (free public API) ─────────────
def fetch_weather(city: str, meta: dict, start: str, end: str) -> pd.DataFrame:
    """
    Calls the Open-Meteo historical weather API for a given city and date range.
    Returns a DataFrame with date, temperature_max, humidity columns.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":        meta["lat"],
        "longitude":       meta["lon"],
        "start_date":      start,
        "end_date":        end,
        "daily":           "temperature_2m_max,relative_humidity_2m_max,weathercode",
        "timezone":        "Asia/Kolkata",
    }
    print(f"  Fetching weather for {city} ({start} → {end}) ...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["daily"]
    df = pd.DataFrame({
        "date":              data["time"],
        "temperature":       data["temperature_2m_max"],
        "humidity":          data["relative_humidity_2m_max"],
        "weather_code":      data["weathercode"],
    })
    df["location"] = city
    df["population"] = meta["population"]
    return df

# ── 3. Pull data for all cities ───────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"   # ~2 years of daily data

all_weather = []
for city, meta in CITIES.items():
    try:
        df_city = fetch_weather(city, meta, START_DATE, END_DATE)
        all_weather.append(df_city)
        time.sleep(0.5)   # polite delay between API calls
    except Exception as e:
        print(f"  WARNING: Could not fetch data for {city}: {e}")

weather_df = pd.concat(all_weather, ignore_index=True)
print(f"\nRaw API rows fetched: {len(weather_df)}")

# ── 4. Clean API data ─────────────────────────────────────────────────────────
# Fill any missing values with city-level medians
weather_df["temperature"] = weather_df.groupby("location")["temperature"].transform(
    lambda x: x.fillna(x.median())
)
weather_df["humidity"] = weather_df.groupby("location")["humidity"].transform(
    lambda x: x.fillna(x.median())
)
weather_df["weather_code"] = weather_df["weather_code"].fillna(0).astype(int)

# Map WMO codes to readable conditions
weather_df["weather_condition"] = weather_df["weather_code"].map(WMO_MAP).fillna("Clear")
weather_df.drop(columns=["weather_code"], inplace=True)

# ── 5. Expand to 50,000–60,000 rows by crossing with sectors AND zones ────────
# Each (city, date) row × 4 sectors × 5 zones = 20× expansion
# 18 cities × 730 days × 4 sectors × ~1 zone sample = ~52,560 rows
weather_df["_key"] = 1
sectors_df = pd.DataFrame({"sector": SECTORS, "_key": 1})
zones_df   = pd.DataFrame({"zone": ZONES,    "_key": 1})

expanded_df = (
    weather_df
    .merge(sectors_df, on="_key")
    .merge(zones_df,   on="_key")
    .drop(columns=["_key"])
)
print(f"Rows after sector+zone expansion: {len(expanded_df)}")

# ── 6. Simulate pollution_index (AQI proxy) ───────────────────────────────────
rng = np.random.default_rng(seed=42)

# Base AQI per city (realistic Indian city AQI ranges)
CITY_AQI_BASE = {
    "Bangalore": 85,  "Hyderabad": 95,   "Chennai": 80,
    "Mumbai":   110,  "Delhi":     180,   "Kolkata": 130,
    "Pune":      90,  "Ahmedabad": 120,   "Jaipur":  115,
    "Surat":    105,  "Lucknow":   160,   "Nagpur":   95,
    "Indore":   100,  "Bhopal":     90,   "Visakhapatnam": 75,
    "Patna":    155,  "Coimbatore": 70,   "Kochi":    65,
}
expanded_df["pollution_index"] = expanded_df["location"].map(CITY_AQI_BASE)

# Add seasonal + random noise to AQI
n = len(expanded_df)
expanded_df["pollution_index"] = (
    expanded_df["pollution_index"]
    + rng.normal(0, 15, n)                          # daily random variation
    + (expanded_df["humidity"] - 60) * 0.3          # higher humidity → more particulates
).clip(20, 500).round(1)

# ── 7. Generate waste_amount (TARGET column) ──────────────────────────────────
# Formula:
#   waste = (population * 0.00005)          ← city-scale base
#         + (temperature * rand_factor)     ← heat increases waste activity
#         + (humidity * small_factor)       ← moisture effect
#         + sector_weight                   ← sector contribution
#         + noise                           ← realistic daily variation
#
# Ensures: Industrial > Commercial > Residential > Healthcare

sector_weight_arr = expanded_df["sector"].map(SECTOR_WEIGHTS).values

# Zone modifier: Central zones generate slightly more waste (denser activity)
ZONE_MODIFIERS = {"North": 1.0, "South": 0.95, "East": 1.05, "West": 0.98, "Central": 1.10}
zone_mod_arr = expanded_df["zone"].map(ZONE_MODIFIERS).values

rand_temp_factor  = rng.uniform(0.3, 0.7, n)
rand_humidity_fac = rng.uniform(0.05, 0.15, n)
noise             = rng.normal(0, 20, n)

expanded_df["waste_amount"] = (
    (expanded_df["population"].values * 0.00005)
    + (expanded_df["temperature"].values * rand_temp_factor)
    + (expanded_df["humidity"].values   * rand_humidity_fac)
    + sector_weight_arr * zone_mod_arr
    + noise
).clip(50, 1000).round(2)

# ── 8. Final column selection & type enforcement ──────────────────────────────
final_df = expanded_df[[
    "location", "date", "temperature", "humidity",
    "weather_condition", "population", "zone", "sector",
    "pollution_index", "waste_amount"
]].copy()

final_df["date"]        = pd.to_datetime(final_df["date"]).dt.strftime("%Y-%m-%d")
final_df["temperature"] = final_df["temperature"].round(1)
final_df["humidity"]    = final_df["humidity"].astype(int)
final_df["population"]  = final_df["population"].astype(int)

# Shuffle rows for ML readiness
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── 9. Validate ───────────────────────────────────────────────────────────────
assert final_df.isnull().sum().sum() == 0, "Dataset contains missing values!"
assert len(final_df) >= 50_000,           "Dataset has fewer than 50,000 rows!"

# ── 10. Save to CSV ───────────────────────────────────────────────────────────
output_path = "waste_dataset.csv"
final_df.to_csv(output_path, index=False)
print(f"\nDataset saved to: {output_path}")

# ── 11. Summary ───────────────────────────────────────────────────────────────
print(f"\nDataset shape: {final_df.shape}")
print(f"Total rows:    {len(final_df):,}  ({'✓ meets 50k+ requirement' if len(final_df) >= 50000 else '✗ below 50k'})")
print(f"\nFirst 5 rows:")
print(final_df.head().to_string(index=False))
print(f"\nColumn dtypes:")
print(final_df.dtypes)
print(f"\nwaste_amount stats:")
print(final_df["waste_amount"].describe().round(2))
