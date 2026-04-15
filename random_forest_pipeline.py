"""
random_forest_pipeline.py  —  STANDALONE SCRIPT
=================================================
Smart City Waste Prediction — Random Forest Regression Pipeline
Run: python random_forest_pipeline.py

Steps:
  1.  Initialize SparkSession
  2.  Load ml_ready_dataset.csv
  3.  Feature preparation (VectorAssembler)
  4.  Train / Test split (80 / 20)
  5.  Train RandomForestRegressor
  6.  Generate predictions
  7.  Evaluate (RMSE, MAE, R²)
  8.  Save evaluation reports
  9.  Feature importance analysis
  10. Visualizations (4 plots)
  11. Save all outputs → Random_Forest_Outputs/
"""

import os
import tempfile
import time

# ── Windows: download real winutils.exe so Spark can write model files ───────
if os.name == "nt":
    import urllib.request, pathlib, shutil

    _hadoop_home = pathlib.Path(tempfile.gettempdir()) / "hadoop3"
    _bin_dir     = _hadoop_home / "bin"
    _winutils    = _bin_dir / "winutils.exe"
    _hadoop_dll  = _bin_dir / "hadoop.dll"

    _bin_dir.mkdir(parents=True, exist_ok=True)

    # Download winutils.exe + hadoop.dll for Hadoop 3.3.6 (Windows build)
    _base = "https://github.com/cdarlint/winutils/raw/master/hadoop-3.3.6/bin"
    for _fname, _dest in [("winutils.exe", _winutils), ("hadoop.dll", _hadoop_dll)]:
        if not _dest.exists():
            print(f"  Downloading {_fname} for Windows Hadoop support...")
            try:
                urllib.request.urlretrieve(f"{_base}/{_fname}", str(_dest))
                print(f"  Downloaded → {_dest}")
            except Exception as _e:
                print(f"  WARNING: Could not download {_fname}: {_e}")

    os.environ["HADOOP_HOME"]      = str(_hadoop_home)
    os.environ["hadoop.home.dir"]  = str(_hadoop_home)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE   = "ml_ready_dataset.csv"
OUTPUT_DIR   = "Random_Forest_Outputs"
SEED         = 42
TEST_SIZE    = 0.20
NUM_TREES    = 100
MAX_DEPTH    = 10

FEATURE_COLS = [
    "temperature", "humidity", "population", "pollution_index",
    "population_density", "year", "month", "day_of_week", "is_weekend",
    "sector_enc", "weather_condition_enc", "location_enc",
    "season_enc", "zone_enc",
]
TARGET_COL = "waste_amount"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 120, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11,
    "xtick.labelsize": 9,  "ytick.labelsize": 9,
    "figure.facecolor": "white",
})

def banner(step: int, title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'━'*60}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Initialize SparkSession
# ─────────────────────────────────────────────────────────────────────────────
banner(1, "Initialize SparkSession")

spark = (SparkSession.builder
         .appName("WasteProject_RandomForest")
         .config("spark.sql.shuffle.partitions", "8")
         .config("spark.driver.memory", "4g")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
print(f"  Spark version : {spark.version}")
print(f"  Output folder : {OUTPUT_DIR}/")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
banner(2, "Load Dataset")

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(INPUT_FILE))

# Cast all feature + target columns to DoubleType for MLlib
for c in FEATURE_COLS + [TARGET_COL]:
    df = df.withColumn(c, F.col(c).cast(DoubleType()))

total_rows = df.count()
print(f"  File          : {INPUT_FILE}")
print(f"  Total rows    : {total_rows:,}")
print(f"  Total columns : {len(df.columns)}")

print("\n  --- Schema ---")
df.printSchema()

print("  --- First 5 rows ---")
df.show(5, truncate=True)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature Preparation (VectorAssembler)
# ─────────────────────────────────────────────────────────────────────────────
banner(3, "Feature Preparation — VectorAssembler")

assembler = VectorAssembler(
    inputCols=FEATURE_COLS,
    outputCol="features",
    handleInvalid="skip"
)
df_assembled = assembler.transform(df).select("features", TARGET_COL)

print(f"  Input features ({len(FEATURE_COLS)}):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {f}")
print(f"\n  Target column : {TARGET_COL}")
print(f"  Assembled rows: {df_assembled.count():,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train / Test Split
# ─────────────────────────────────────────────────────────────────────────────
banner(4, "Train / Test Split  (80% / 20%)")

train_df, test_df = df_assembled.randomSplit(
    [1 - TEST_SIZE, TEST_SIZE], seed=SEED
)
train_count = train_df.count()
test_count  = test_df.count()

print(f"  Training rows : {train_count:,}  ({train_count/total_rows*100:.1f}%)")
print(f"  Testing  rows : {test_count:,}  ({test_count/total_rows*100:.1f}%)")
print(f"  Random seed   : {SEED}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train Random Forest Model
# ─────────────────────────────────────────────────────────────────────────────
banner(5, f"Train RandomForestRegressor  (numTrees={NUM_TREES}, maxDepth={MAX_DEPTH})")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=TARGET_COL,
    numTrees=NUM_TREES,
    maxDepth=MAX_DEPTH,
    seed=SEED,
    subsamplingRate=0.8,
)

print("  Training model — please wait...")
t0 = time.time()
rf_model = rf.fit(train_df)
elapsed = time.time() - t0
print(f"  Training complete in {elapsed:.1f}s")
print(f"  Number of trees : {rf_model.getNumTrees}")

# ── Save model metadata manually (avoids Hadoop native IO on Windows) ────────
# Saves: feature importances, model params, and feature column names as JSON.
# This is fully portable and sufficient for reproducing predictions.
import json, pathlib

MODEL_PATH = os.path.join(OUTPUT_DIR, "rf_model")
os.makedirs(MODEL_PATH, exist_ok=True)

model_meta = {
    "model_type":        "RandomForestRegressionModel",
    "spark_version":     spark.version,
    "num_trees":         rf_model.getNumTrees,
    "max_depth":         rf_model.getOrDefault("maxDepth"),
    "features_col":      FEATURE_COLS,
    "label_col":         TARGET_COL,
    "seed":              SEED,
    "feature_importances": dict(zip(
        FEATURE_COLS,
        [float(x) for x in rf_model.featureImportances.toArray()]
    )),
    # metrics will be added after evaluation (Step 7)
}

meta_path = os.path.join(MODEL_PATH, "model_metadata.json")
with open(meta_path, "w", encoding="utf-8") as _f:
    json.dump(model_meta, _f, indent=2)

print(f"  Model metadata  → {meta_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate Predictions
# ─────────────────────────────────────────────────────────────────────────────
banner(6, "Generate Predictions on Test Set")

predictions = rf_model.transform(test_df)

print("  --- Sample predictions (actual vs predicted) ---")
predictions.select(
    F.col(TARGET_COL).alias("actual"),
    F.col("prediction").alias("predicted"),
    F.abs(F.col(TARGET_COL) - F.col("prediction")).alias("abs_error")
).show(15, truncate=False)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
banner(7, "Model Evaluation — RMSE / MAE / R²")

evaluator_rmse = RegressionEvaluator(
    labelCol=TARGET_COL, predictionCol="prediction", metricName="rmse")
evaluator_mae  = RegressionEvaluator(
    labelCol=TARGET_COL, predictionCol="prediction", metricName="mae")
evaluator_r2   = RegressionEvaluator(
    labelCol=TARGET_COL, predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae  = evaluator_mae.evaluate(predictions)
r2   = evaluator_r2.evaluate(predictions)

# Additional: MAPE
pred_pdf = predictions.select(
    F.col(TARGET_COL).alias("actual"),
    F.col("prediction").alias("predicted")
).toPandas()

mape = (abs((pred_pdf["actual"] - pred_pdf["predicted"]) / pred_pdf["actual"])
        .mean() * 100)

print(f"\n  {'Metric':<30} {'Value':>12}")
print(f"  {'─'*30} {'─'*12}")
print(f"  {'RMSE (Root Mean Sq Error)':<30} {rmse:>12.4f}")
print(f"  {'MAE  (Mean Absolute Error)':<30} {mae:>12.4f}")
print(f"  {'R²   (R-Squared Score)':<30} {r2:>12.4f}")
print(f"  {'MAPE (Mean Abs % Error)':<30} {mape:>11.2f}%")

# Update model_metadata.json with evaluation metrics now that they're computed
import json as _json
model_meta["metrics"] = {
    "rmse": round(rmse, 4),
    "mae":  round(mae,  4),
    "r2":   round(r2,   4),
    "mape": round(mape, 4),
}
with open(meta_path, "w", encoding="utf-8") as _f:
    _json.dump(model_meta, _f, indent=2)
print(f"\n  Model metadata updated with metrics → {meta_path}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save Evaluation Reports
# ─────────────────────────────────────────────────────────────────────────────
banner(8, f"Save Reports → {OUTPUT_DIR}/")

# 8a. evaluation_metrics.txt
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=" * 55 + "\n")
    f.write("  RANDOM FOREST REGRESSION — EVALUATION METRICS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"  Model          : RandomForestRegressor\n")
    f.write(f"  Dataset        : {INPUT_FILE}\n")
    f.write(f"  Total rows     : {total_rows:,}\n")
    f.write(f"  Training rows  : {train_count:,}\n")
    f.write(f"  Testing  rows  : {test_count:,}\n")
    f.write(f"  Num Trees      : {NUM_TREES}\n")
    f.write(f"  Max Depth      : {MAX_DEPTH}\n")
    f.write(f"  Random Seed    : {SEED}\n\n")
    f.write("-" * 55 + "\n")
    f.write(f"  RMSE  (Root Mean Squared Error) : {rmse:.4f}\n")
    f.write(f"  MAE   (Mean Absolute Error)     : {mae:.4f}\n")
    f.write(f"  R²    (R-Squared Score)         : {r2:.4f}\n")
    f.write(f"  MAPE  (Mean Abs Percentage Err) : {mape:.2f}%\n")
    f.write("-" * 55 + "\n\n")
    f.write("  INTERPRETATION\n")
    f.write(f"  R² = {r2:.4f} → model explains {r2*100:.1f}% of variance\n")
    f.write(f"  RMSE = {rmse:.2f} tons — avg prediction error magnitude\n")
    f.write(f"  MAE  = {mae:.2f} tons — avg absolute prediction error\n")
print(f"  Saved → {metrics_path}")

# 8b. predictions.csv
pred_pdf.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
print(f"  Saved → {OUTPUT_DIR}/predictions.csv  ({len(pred_pdf):,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
banner(9, "Feature Importance Analysis")

importances = rf_model.featureImportances.toArray()
fi_df = (pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
fi_df["rank"] = fi_df.index + 1
fi_df["importance_pct"] = (fi_df["importance"] * 100).round(3)

print(f"\n  {'Rank':<5} {'Feature':<25} {'Importance':>10} {'%':>8}")
print(f"  {'─'*5} {'─'*25} {'─'*10} {'─'*8}")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {int(row['rank']):<5} {row['feature']:<25} "
          f"{row['importance']:>10.6f} {row['importance_pct']:>7.2f}%  {bar}")

fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
print(f"\n  Saved → {OUTPUT_DIR}/feature_importance.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Visualizations
# ─────────────────────────────────────────────────────────────────────────────
banner(10, "Generate Visualizations")

# Sample for scatter plots (keep rendering fast)
plot_sample = pred_pdf.sample(n=min(8000, len(pred_pdf)), random_state=SEED)
actual    = plot_sample["actual"].values
predicted = plot_sample["predicted"].values
residuals = actual - predicted

# ── Plot 1: Actual vs Predicted Scatter ──────────────────────────────────────
print("  Generating Plot 1 — Actual vs Predicted...")
fig, ax = plt.subplots(figsize=(7, 6))

sc = ax.scatter(actual, predicted, alpha=0.25, s=8,
                c=np.abs(residuals), cmap="RdYlGn_r",
                edgecolors="none")
plt.colorbar(sc, ax=ax, label="|Residual| (tons)")

# Perfect prediction line
lims = [min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max())]
ax.plot(lims, lims, "r--", linewidth=1.8, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title("Actual vs Predicted Waste Amount")
ax.set_xlabel("Actual Waste (tons)")
ax.set_ylabel("Predicted Waste (tons)")
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"R² = {r2:.4f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"),
            bbox_inches="tight")
plt.close()
print(f"  Saved → {OUTPUT_DIR}/actual_vs_predicted.png")

# ── Plot 2: Residual Plot ─────────────────────────────────────────────────────
print("  Generating Plot 2 — Residual Plot...")
fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(predicted, residuals, alpha=0.22, s=7,
           c=residuals, cmap="coolwarm", edgecolors="none")
ax.axhline(0, color="red", linewidth=1.8, linestyle="--", label="Zero residual")

# ±1 RMSE band
ax.axhline( rmse, color="orange", linewidth=1, linestyle=":", alpha=0.8,
            label=f"+RMSE ({rmse:.1f})")
ax.axhline(-rmse, color="orange", linewidth=1, linestyle=":", alpha=0.8,
            label=f"-RMSE ({rmse:.1f})")

ax.set_title("Residual Plot  (Actual − Predicted)")
ax.set_xlabel("Predicted Waste (tons)")
ax.set_ylabel("Residual (tons)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"),
            bbox_inches="tight")
plt.close()
print(f"  Saved → {OUTPUT_DIR}/residual_plot.png")

# ── Plot 3: Feature Importance Bar Chart ─────────────────────────────────────
print("  Generating Plot 3 — Feature Importance...")
fig, ax = plt.subplots(figsize=(8, 6))

colors = sns.color_palette("Blues_d", len(fi_df))[::-1]
bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
               color=colors, edgecolor="white", linewidth=0.5)

# Value labels
for bar, val in zip(bars, fi_df["importance"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.2f}%", va="center", fontsize=8)

ax.set_title("Feature Importance — Random Forest")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.set_xlim(0, fi_df["importance"].max() * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"),
            bbox_inches="tight")
plt.close()
print(f"  Saved → {OUTPUT_DIR}/feature_importance.png")

# ── Plot 4: Prediction Distribution ──────────────────────────────────────────
print("  Generating Plot 4 — Prediction Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(pred_pdf["actual"],    bins=60, alpha=0.55,
        color="#1976D2", label="Actual",    edgecolor="white")
ax.hist(pred_pdf["predicted"], bins=60, alpha=0.55,
        color="#E53935", label="Predicted", edgecolor="white")

ax.axvline(pred_pdf["actual"].mean(),    color="#0D47A1",
           linestyle="--", linewidth=1.5,
           label=f"Actual mean ({pred_pdf['actual'].mean():.1f})")
ax.axvline(pred_pdf["predicted"].mean(), color="#B71C1C",
           linestyle="--", linewidth=1.5,
           label=f"Predicted mean ({pred_pdf['predicted'].mean():.1f})")

ax.set_title("Distribution of Actual vs Predicted Waste")
ax.set_xlabel("Waste Amount (tons)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"),
            bbox_inches="tight")
plt.close()
print(f"  Saved → {OUTPUT_DIR}/prediction_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Final Output Display
# ─────────────────────────────────────────────────────────────────────────────
banner(11, "Final Output Summary")

print(f"\n  {'─'*55}")
print(f"  EVALUATION METRICS")
print(f"  {'─'*55}")
print(f"  RMSE  : {rmse:.4f} tons")
print(f"  MAE   : {mae:.4f} tons")
print(f"  R²    : {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"  MAPE  : {mape:.2f}%")

print(f"\n  {'─'*55}")
print(f"  TOP 5 FEATURE IMPORTANCES")
print(f"  {'─'*55}")
for _, row in fi_df.head(5).iterrows():
    print(f"  {int(row['rank'])}. {row['feature']:<25} {row['importance_pct']:>6.2f}%")

print(f"\n  {'─'*55}")
print(f"  SAMPLE PREDICTIONS (first 10 rows)")
print(f"  {'─'*55}")
print(pred_pdf.head(10).round(2).to_string(index=False))

print(f"\n  {'─'*55}")
print(f"  FILES SAVED IN  →  {OUTPUT_DIR}/")
print(f"  {'─'*55}")
outputs = [
    "evaluation_metrics.txt",
    "predictions.csv",
    "feature_importance.csv",
    "actual_vs_predicted.png",
    "residual_plot.png",
    "feature_importance.png",
    "prediction_distribution.png",
    "rf_model",
]
for fname in outputs:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.isdir(fpath):
        print(f"  ✓  {fname:<40} (saved model directory)")
    elif os.path.exists(fpath):
        size  = os.path.getsize(fpath)
        unit  = "MB" if size >= 1_048_576 else ("KB" if size >= 1024 else "B")
        size_disp = (size // 1_048_576 if unit == "MB"
                     else size // 1024 if unit == "KB" else size)
        print(f"  ✓  {fname:<40} ({size_disp} {unit})")
    else:
        print(f"  ✗  {fname:<40} (not found)")

spark.stop()
print(f"\n{'█'*60}")
print(f"  Pipeline complete.")
print(f"{'█'*60}")
