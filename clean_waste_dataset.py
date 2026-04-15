"""
Smart City Waste Prediction — Data Cleaning & Feature Engineering Pipeline
--------------------------------------------------------------------------
Input : waste_dataset.csv
Output: waste_dataset_cleaned.csv

Steps:
  1. Load & inspect
  2. Fix date format (DD-MM-YYYY → YYYY-MM-DD) + parse
  3. Extract time-based features (year, month, day, day_of_week, is_weekend)
  4. Add season feature
  5. Fix capped waste_amount values (replace 1000.0 with realistic 800–1200)
  6. Data quality checks (nulls, dtypes, duplicates)
  7. Add population_density feature
  8. Label-encode categorical columns (sector, weather_condition, location, season)
  9. Save cleaned CSV
"""

import pandas as pd
import numpy as np

rng = np.random.default_rng(seed=99)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv("waste_dataset.csv")
print(f"Shape BEFORE cleaning: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ── 2. Fix & parse date (DD-MM-YYYY → YYYY-MM-DD) ────────────────────────────
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")   # keep a clean string col too

# ── 3. Time-based features ────────────────────────────────────────────────────
df["year"]        = df["date"].dt.year
df["month"]       = df["date"].dt.month
df["day"]         = df["date"].dt.day
df["day_of_week"] = df["date"].dt.dayofweek          # 0=Monday … 6=Sunday
df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)

# ── 4. Season feature ─────────────────────────────────────────────────────────
def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    elif month in (3, 4, 5):
        return "Summer"
    elif month in (6, 7, 8, 9):
        return "Rainy"
    else:                          # 10, 11
        return "Post-Monsoon"

df["season"] = df["month"].apply(get_season)

# ── 5. Fix capped waste_amount ────────────────────────────────────────────────
# 36,682 rows are hard-capped at 1000.0 due to .clip(50, 1000) in generation.
# Replace them with realistic values in [800, 1200] using controlled randomness.
cap_mask = df["waste_amount"] == 1000.0
n_capped = cap_mask.sum()
print(f"Capped waste_amount rows (== 1000): {n_capped:,}")

# Draw replacements from a truncated normal centred at 1050 (slightly above old cap)
# so the distribution is continuous and no extreme outliers appear.
replacements = rng.normal(loc=1050, scale=80, size=n_capped)
replacements = np.clip(replacements, 800, 1200).round(2)
df.loc[cap_mask, "waste_amount"] = replacements

print(f"waste_amount range after fix: {df['waste_amount'].min():.2f} – {df['waste_amount'].max():.2f}")

# ── 6. Data quality ───────────────────────────────────────────────────────────
# 6a. Null check
assert df.isnull().sum().sum() == 0, "Nulls found — investigate!"
print("Null check: PASSED")

# 6b. Duplicates
before_dedup = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {before_dedup - len(df)}")

# 6c. Enforce dtypes
df["temperature"]     = df["temperature"].astype("float32")
df["humidity"]        = df["humidity"].astype("int16")
df["pollution_index"] = df["pollution_index"].astype("float32")
df["waste_amount"]    = df["waste_amount"].astype("float32")
df["population"]      = df["population"].astype("int64")
df["year"]            = df["year"].astype("int16")
df["month"]           = df["month"].astype("int8")
df["day"]             = df["day"].astype("int8")
df["day_of_week"]     = df["day_of_week"].astype("int8")
df["is_weekend"]      = df["is_weekend"].astype("int8")

# ── 7. Population density feature ────────────────────────────────────────────
# Approximate city areas in km² (realistic values from public data)
CITY_AREA_KM2 = {
    "Bangalore":     741,   "Hyderabad":     650,   "Chennai":       426,
    "Mumbai":        603,   "Delhi":        1484,   "Kolkata":       185,
    "Pune":          331,   "Ahmedabad":     505,   "Jaipur":        467,
    "Surat":         395,   "Lucknow":       349,   "Nagpur":        227,
    "Indore":        530,   "Bhopal":        463,   "Visakhapatnam": 682,
    "Patna":         136,   "Coimbatore":    246,   "Kochi":          94,
}
df["area_km2"]           = df["location"].map(CITY_AREA_KM2).astype("float32")
df["population_density"] = (df["population"] / df["area_km2"]).round(2).astype("float32")

# ── 8. Label-encode categorical columns ──────────────────────────────────────
# Using pandas Categorical codes — Spark-friendly integer indices.
# Mapping tables are printed so they can be reused for inference.

categorical_cols = ["sector", "weather_condition", "location", "season", "zone"]
label_maps = {}

for col in categorical_cols:
    cat = pd.Categorical(df[col])
    df[f"{col}_enc"] = cat.codes.astype("int16")
    label_maps[col] = dict(enumerate(cat.categories))
    print(f"\n{col} encoding map:")
    for idx, val in label_maps[col].items():
        print(f"  {idx} → {val}")

# ── 9. Final column order ─────────────────────────────────────────────────────
# Keep both raw strings (for interpretability) and encoded ints (for ML)
final_cols = [
    # identifiers / raw
    "location", "date_str",
    # weather features
    "temperature", "humidity", "weather_condition",
    # city features
    "population", "area_km2", "population_density",
    # zone / sector (raw + encoded)
    "zone", "zone_enc",
    "sector", "sector_enc",
    # pollution
    "pollution_index",
    # time features
    "year", "month", "day", "day_of_week", "is_weekend",
    # season (raw + encoded)
    "season", "season_enc",
    # encoded weather & location
    "weather_condition_enc", "location_enc",
    # TARGET
    "waste_amount",
]

df = df[final_cols].rename(columns={"date_str": "date"})

# ── 10. Final validation ──────────────────────────────────────────────────────
assert df.isnull().sum().sum() == 0, "Nulls in final dataset!"
assert len(df) >= 50_000,            "Row count dropped below 50k!"

# ── 11. Save ──────────────────────────────────────────────────────────────────
df.to_csv("waste_dataset_cleaned.csv", index=False)
print(f"\nSaved → waste_dataset_cleaned.csv")

# ── 12. Summary ───────────────────────────────────────────────────────────────
print(f"\nShape AFTER cleaning : {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst 5 rows:")
print(df.head().to_string(index=False))
print(f"\nwaste_amount distribution:")
print(df["waste_amount"].describe().round(2))
