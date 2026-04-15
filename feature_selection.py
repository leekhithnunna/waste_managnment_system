"""
feature_selection.py  —  STANDALONE SCRIPT
===========================================
Run: python feature_selection.py
Requires: cleaned_dataset.csv  (produced by data_cleaning.py)

Tasks:
  1. Start SparkSession
  2. Load cleaned_dataset.csv
  3. Select only ML-relevant feature columns
  4. Print selected columns and dataset shape
  5. Save → ml_ready_dataset.csv
"""

import os
import tempfile

# ── Windows: stub HADOOP_HOME ─────────────────────────────────────────────────
if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
    _stub = os.path.join(tempfile.gettempdir(), "hadoop_stub")
    os.makedirs(os.path.join(_stub, "bin"), exist_ok=True)
    os.environ["HADOOP_HOME"] = _stub

from pyspark.sql import SparkSession

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "cleaned_dataset.csv"
OUTPUT_FILE = "ml_ready_dataset.csv"

# Columns to KEEP for ML training (encoded + numeric + target)
ML_FEATURES = [
    "temperature",
    "humidity",
    "population",
    "pollution_index",
    "population_density",
    "year",
    "month",
    "day_of_week",
    "is_weekend",
    "sector_enc",
    "weather_condition_enc",
    "location_enc",
    "season_enc",
    "zone_enc",
    "waste_amount",           # TARGET
]

# Columns being removed (raw categoricals + non-ML fields)
REMOVED_COLS = [
    "sector", "weather_condition", "location",
    "season", "date", "area_km2", "zone", "day",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SparkSession
# ─────────────────────────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("WasteProject_FeatureSelection")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("SparkSession started.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load cleaned dataset
# ─────────────────────────────────────────────────────────────────────────────
df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(INPUT_FILE))

total_rows = df.count()
total_cols = len(df.columns)

print(f"\n{'='*55}")
print(f"  Loaded : {INPUT_FILE}")
print(f"  Shape  : {total_rows:,} rows × {total_cols} cols")
print(f"{'='*55}")

print("\n--- All available columns ---")
for i, c in enumerate(df.columns, 1):
    print(f"  {i:>2}. {c}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Select ML features
# ─────────────────────────────────────────────────────────────────────────────
df_ml = df.select(ML_FEATURES)

print(f"\n--- Columns REMOVED ({len(REMOVED_COLS)}) ---")
for c in REMOVED_COLS:
    print(f"  ✗  {c}")

print(f"\n--- Columns KEPT ({len(ML_FEATURES)}) ---")
for i, c in enumerate(ML_FEATURES, 1):
    marker = "← TARGET" if c == "waste_amount" else ""
    print(f"  {i:>2}. {c}  {marker}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Print shape + preview
# ─────────────────────────────────────────────────────────────────────────────
ml_rows = df_ml.count()
ml_cols = len(df_ml.columns)

print(f"\n{'='*55}")
print(f"  FEATURE SELECTION SUMMARY")
print(f"{'='*55}")
print(f"  Before : {total_rows:,} rows × {total_cols} cols")
print(f"  After  : {ml_rows:,} rows × {ml_cols} cols")
print(f"  Cols removed : {total_cols - ml_cols}")
print(f"{'='*55}")

print("\n--- First 5 rows (ML-ready) ---")
df_ml.show(5, truncate=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save to CSV (via pandas)
# ─────────────────────────────────────────────────────────────────────────────
pdf = df_ml.toPandas()
pdf.to_csv(OUTPUT_FILE, index=False)
print(f"Saved → {OUTPUT_FILE}  ({len(pdf):,} rows × {len(pdf.columns)} cols)")

spark.stop()
print("\nSparkSession stopped. feature_selection.py complete.")
