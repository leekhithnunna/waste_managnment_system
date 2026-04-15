"""
data_cleaning.py  —  STANDALONE SCRIPT
=======================================
Run: python data_cleaning.py

Tasks:
  1. Start SparkSession
  2. Load waste_dataset_cleaned.csv
  3. Print schema, first 5 rows, row count
  4. Cast numeric columns to correct types
  5. Parse date column
  6. Remove duplicates
  7. Drop null rows
  8. Validate (assert zero nulls)
  9. Save → cleaned_dataset.csv
 10. Print shape before / after
"""

import os
import tempfile

# ── Windows: stub HADOOP_HOME so Spark can write local files ─────────────────
if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
    _stub = os.path.join(tempfile.gettempdir(), "hadoop_stub")
    os.makedirs(os.path.join(_stub, "bin"), exist_ok=True)
    os.environ["HADOOP_HOME"] = _stub

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "waste_dataset_cleaned.csv"
OUTPUT_FILE = "cleaned_dataset.csv"

DOUBLE_COLS = ["temperature", "pollution_index",
               "population_density", "area_km2", "waste_amount"]

INT_COLS    = ["humidity", "population", "zone_enc", "sector_enc",
               "year", "month", "day", "day_of_week", "is_weekend",
               "season_enc", "weather_condition_enc", "location_enc"]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SparkSession
# ─────────────────────────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("WasteProject_Cleaning")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print("SparkSession started.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load dataset
# ─────────────────────────────────────────────────────────────────────────────
df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(INPUT_FILE))

rows_before = df.count()
cols_before = len(df.columns)

print(f"\n{'='*55}")
print(f"  Loaded : {INPUT_FILE}")
print(f"  Shape  : {rows_before:,} rows × {cols_before} cols")
print(f"{'='*55}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Schema + preview
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Schema ---")
df.printSchema()

print("--- First 5 rows ---")
df.show(5, truncate=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Cast numeric columns
# ─────────────────────────────────────────────────────────────────────────────
for c in DOUBLE_COLS:
    df = df.withColumn(c, F.col(c).cast(DoubleType()))
for c in INT_COLS:
    df = df.withColumn(c, F.col(c).cast(IntegerType()))
print("\n[1/4] Numeric columns cast to correct types.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Parse date column
# ─────────────────────────────────────────────────────────────────────────────
df = df.withColumn("date", F.to_date(F.col("date"), "yyyy-MM-dd"))
print("[2/4] Date column parsed → DateType (yyyy-MM-dd).")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Remove duplicates
# ─────────────────────────────────────────────────────────────────────────────
before_dedup = df.count()
df = df.dropDuplicates()
after_dedup  = df.count()
print(f"[3/4] Duplicates removed : {before_dedup - after_dedup:,}  "
      f"({before_dedup:,} → {after_dedup:,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Drop null rows
# ─────────────────────────────────────────────────────────────────────────────
before_null = df.count()
df = df.dropna()
after_null  = df.count()
print(f"[4/4] Null rows dropped  : {before_null - after_null:,}  "
      f"({before_null:,} → {after_null:,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Validate
# ─────────────────────────────────────────────────────────────────────────────
null_counts = df.select(
    [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]
).collect()[0].asDict()

total_nulls = sum(null_counts.values())
assert total_nulls == 0, f"VALIDATION FAILED — {total_nulls} nulls remain!"
print(f"\nValidation PASSED — zero nulls confirmed.")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save to CSV (via pandas to avoid winutils on Windows)
# ─────────────────────────────────────────────────────────────────────────────
pdf = df.toPandas()
pdf.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Shape summary
# ─────────────────────────────────────────────────────────────────────────────
rows_after = len(pdf)
cols_after = len(pdf.columns)

print(f"\n{'='*55}")
print(f"  SHAPE SUMMARY")
print(f"{'='*55}")
print(f"  Before cleaning : {rows_before:,} rows × {cols_before} cols")
print(f"  After  cleaning : {rows_after:,} rows × {cols_after} cols")
print(f"  Rows removed    : {rows_before - rows_after:,}")
print(f"{'='*55}")

print("\n--- Descriptive statistics (key columns) ---")
print(pdf[["temperature", "humidity", "pollution_index", "waste_amount"]]
      .describe().round(2).to_string())

spark.stop()
print("\nSparkSession stopped. data_cleaning.py complete.")
