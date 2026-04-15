"""
xgboost_pipeline.py  —  STANDALONE SCRIPT
==========================================
Smart City Waste Prediction — XGBoost Regression Pipeline

Mirrors random_forest_pipeline.py exactly:
  - Same dataset, same features, same target
  - Same train/test split (80/20, seed=42)
  - Same evaluation metrics (RMSE, MAE, R², MAPE)
  - Same output folder structure
  - Same visualisation plots

Run: python xgboost_pipeline.py
"""

import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (mirror random_forest_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE  = "ml_ready_dataset.csv"
OUTPUT_DIR  = "XGBoost_Outputs"
SEED        = 42
TEST_SIZE   = 0.20

FEATURE_COLS = [
    "temperature", "humidity", "population", "pollution_index",
    "population_density", "year", "month", "day_of_week", "is_weekend",
    "sector_enc", "weather_condition_enc", "location_enc",
    "season_enc", "zone_enc",
]
TARGET_COL = "waste_amount"

# ── XGBoost hyperparameters (defaults — easy to tune) ────────────────────────
XGB_PARAMS = {
    "n_estimators":      500,       # number of boosting rounds
    "max_depth":         6,         # max tree depth (default=6)
    "learning_rate":     0.1,       # step size shrinkage (eta)
    "subsample":         0.8,       # row subsampling per tree
    "colsample_bytree":  0.8,       # feature subsampling per tree
    "min_child_weight":  1,         # minimum sum of instance weight in a leaf
    "reg_alpha":         0.0,       # L1 regularisation
    "reg_lambda":        1.0,       # L2 regularisation
    "random_state":      SEED,
    "n_jobs":            -1,        # use all CPU cores
    "verbosity":         0,         # suppress XGBoost internal logs
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":       120,
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.facecolor": "white",
})

def banner(step: int, title: str) -> None:
    print(f"\n{'━'*60}")
    print(f"  STEP {step}: {title}")
    print(f"{'━'*60}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
banner(1, "Load Dataset")

df = pd.read_csv(INPUT_FILE)

print(f"  File          : {INPUT_FILE}")
print(f"  Total rows    : {len(df):,}")
print(f"  Total columns : {len(df.columns)}")
print(f"\n  Columns: {df.columns.tolist()}")
print(f"\n  --- First 5 rows ---")
print(df.head().to_string(index=False))
print(f"\n  --- Data types ---")
print(df.dtypes.to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Data Quality Check
# ─────────────────────────────────────────────────────────────────────────────
banner(2, "Data Quality Check")

# Check for missing values
null_counts = df.isnull().sum()
print(f"  Missing values per column:")
print(null_counts[null_counts > 0].to_string() if null_counts.sum() > 0
      else "  None — dataset is clean.")

# Drop any nulls if present (mirrors RF pipeline's dropna)
before = len(df)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
after = len(df)
print(f"\n  Rows before cleaning : {before:,}")
print(f"  Rows after  cleaning : {after:,}")
print(f"  Rows removed         : {before - after:,}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Feature / Target Split
# ─────────────────────────────────────────────────────────────────────────────
banner(3, "Feature / Target Split")

X = df[FEATURE_COLS]
y = df[TARGET_COL]

print(f"  Input features ({len(FEATURE_COLS)}):")
for i, f in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {f}")
print(f"\n  Target column : {TARGET_COL}")
print(f"  X shape       : {X.shape}")
print(f"  y shape       : {y.shape}")
print(f"\n  Target stats  :")
print(y.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train / Test Split  (80 / 20 — mirrors RF pipeline)
# ─────────────────────────────────────────────────────────────────────────────
banner(4, "Train / Test Split  (80% / 20%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEED,
)

print(f"  Training rows : {len(X_train):,}  ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Testing  rows : {len(X_test):,}  ({len(X_test)/len(df)*100:.1f}%)")
print(f"  Random seed   : {SEED}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Train XGBoost Model
# ─────────────────────────────────────────────────────────────────────────────
banner(5, f"Train XGBRegressor  "
          f"(n_estimators={XGB_PARAMS['n_estimators']}, "
          f"max_depth={XGB_PARAMS['max_depth']}, "
          f"lr={XGB_PARAMS['learning_rate']})")

print("  Hyperparameters:")
for k, v in XGB_PARAMS.items():
    print(f"    {k:<22} = {v}")

model = XGBRegressor(**XGB_PARAMS)

print("\n  Training model — please wait...")
t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
elapsed = time.time() - t0
print(f"  Training complete in {elapsed:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate Predictions
# ─────────────────────────────────────────────────────────────────────────────
banner(6, "Generate Predictions on Test Set")

y_pred = model.predict(X_test)

# Build predictions DataFrame
pred_df = pd.DataFrame({
    "actual":    y_test.values,
    "predicted": y_pred,
    "abs_error": np.abs(y_test.values - y_pred),
})

print("  --- Sample predictions (actual vs predicted) ---")
print(pred_df.head(15).round(2).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────
banner(7, "Model Evaluation — RMSE / MAE / R²")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)
mape = (np.abs((y_test.values - y_pred) / y_test.values).mean()) * 100

print(f"\n  {'Metric':<30} {'Value':>12}")
print(f"  {'─'*30} {'─'*12}")
print(f"  {'RMSE (Root Mean Sq Error)':<30} {rmse:>12.4f}")
print(f"  {'MAE  (Mean Absolute Error)':<30} {mae:>12.4f}")
print(f"  {'R²   (R-Squared Score)':<30} {r2:>12.4f}")
print(f"  {'MAPE (Mean Abs % Error)':<30} {mape:>11.2f}%")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Save Evaluation Reports
# ─────────────────────────────────────────────────────────────────────────────
banner(8, f"Save Reports  →  {OUTPUT_DIR}/")

# 8a. evaluation_metrics.txt
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=" * 55 + "\n")
    f.write("  XGBOOST REGRESSION — EVALUATION METRICS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"  Model          : XGBRegressor\n")
    f.write(f"  Dataset        : {INPUT_FILE}\n")
    f.write(f"  Total rows     : {len(df):,}\n")
    f.write(f"  Training rows  : {len(X_train):,}\n")
    f.write(f"  Testing  rows  : {len(X_test):,}\n")
    f.write(f"  n_estimators   : {XGB_PARAMS['n_estimators']}\n")
    f.write(f"  max_depth      : {XGB_PARAMS['max_depth']}\n")
    f.write(f"  learning_rate  : {XGB_PARAMS['learning_rate']}\n")
    f.write(f"  Random Seed    : {SEED}\n\n")
    f.write("-" * 55 + "\n")
    f.write(f"  RMSE  (Root Mean Squared Error) : {rmse:.4f}\n")
    f.write(f"  MAE   (Mean Absolute Error)     : {mae:.4f}\n")
    f.write(f"  R2    (R-Squared Score)         : {r2:.4f}\n")
    f.write(f"  MAPE  (Mean Abs Percentage Err) : {mape:.2f}%\n")
    f.write("-" * 55 + "\n\n")
    f.write("  INTERPRETATION\n")
    f.write(f"  R2 = {r2:.4f} -> model explains {r2*100:.1f}% of variance\n")
    f.write(f"  RMSE = {rmse:.2f} tons — avg prediction error magnitude\n")
    f.write(f"  MAE  = {mae:.2f} tons — avg absolute prediction error\n")
print(f"  Saved -> {metrics_path}")

# 8b. evaluation_metrics.json
json_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
metrics_dict = {
    "model":         "XGBRegressor",
    "dataset":       INPUT_FILE,
    "total_rows":    len(df),
    "train_rows":    len(X_train),
    "test_rows":     len(X_test),
    "hyperparameters": XGB_PARAMS,
    "metrics": {
        "rmse": round(rmse, 4),
        "mae":  round(mae,  4),
        "r2":   round(r2,   4),
        "mape": round(mape, 4),
    }
}
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(metrics_dict, f, indent=2)
print(f"  Saved -> {json_path}")

# 8c. predictions.csv
pred_csv = os.path.join(OUTPUT_DIR, "predictions.csv")
pred_df[["actual", "predicted"]].to_csv(pred_csv, index=False)
print(f"  Saved -> {pred_csv}  ({len(pred_df):,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Feature Importance Analysis
# ─────────────────────────────────────────────────────────────────────────────
banner(9, "Feature Importance Analysis")

# XGBoost provides three importance types; 'gain' is most informative
importances = model.feature_importances_          # default = 'weight'
fi_df = (pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
           .sort_values("importance", ascending=False)
           .reset_index(drop=True))
fi_df["rank"]           = fi_df.index + 1
fi_df["importance_pct"] = (fi_df["importance"] * 100).round(3)

print(f"\n  {'Rank':<5} {'Feature':<25} {'Importance':>10} {'%':>8}")
print(f"  {'─'*5} {'─'*25} {'─'*10} {'─'*8}")
for _, row in fi_df.iterrows():
    bar = "█" * int(row["importance"] * 50)
    print(f"  {int(row['rank']):<5} {row['feature']:<25} "
          f"{row['importance']:>10.6f} {row['importance_pct']:>7.2f}%  {bar}")

fi_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
fi_df.to_csv(fi_csv, index=False)
print(f"\n  Saved -> {fi_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Visualisations  (mirrors RF pipeline plots)
# ─────────────────────────────────────────────────────────────────────────────
banner(10, "Generate Visualisations")

actual    = pred_df["actual"].values
predicted = pred_df["predicted"].values
residuals = actual - predicted

# Sample for scatter plots
rng = np.random.default_rng(SEED)
idx = rng.choice(len(actual), size=min(8000, len(actual)), replace=False)
act_s, pred_s, res_s = actual[idx], predicted[idx], residuals[idx]

# ── Plot 1: Actual vs Predicted ───────────────────────────────────────────────
print("  Generating Plot 1 — Actual vs Predicted...")
fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(act_s, pred_s, alpha=0.25, s=8,
                c=np.abs(res_s), cmap="RdYlGn_r", edgecolors="none")
plt.colorbar(sc, ax=ax, label="|Residual| (tons)")
lims = [min(act_s.min(), pred_s.min()), max(act_s.max(), pred_s.max())]
ax.plot(lims, lims, "r--", linewidth=1.8, label="Perfect prediction")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_title("Actual vs Predicted Waste Amount  [XGBoost]")
ax.set_xlabel("Actual Waste (tons)")
ax.set_ylabel("Predicted Waste (tons)")
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"R² = {r2:.4f}\nRMSE = {rmse:.2f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "actual_vs_predicted.png"), bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/actual_vs_predicted.png")

# ── Plot 2: Residual Plot ─────────────────────────────────────────────────────
print("  Generating Plot 2 — Residual Plot...")
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pred_s, res_s, alpha=0.22, s=7,
           c=res_s, cmap="coolwarm", edgecolors="none")
ax.axhline(0,     color="red",    linewidth=1.8, linestyle="--", label="Zero residual")
ax.axhline( rmse, color="orange", linewidth=1,   linestyle=":",  alpha=0.8,
            label=f"+RMSE ({rmse:.1f})")
ax.axhline(-rmse, color="orange", linewidth=1,   linestyle=":",  alpha=0.8,
            label=f"-RMSE ({rmse:.1f})")
ax.set_title("Residual Plot  (Actual - Predicted)  [XGBoost]")
ax.set_xlabel("Predicted Waste (tons)")
ax.set_ylabel("Residual (tons)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"), bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/residual_plot.png")

# ── Plot 3: Feature Importance Bar Chart ─────────────────────────────────────
print("  Generating Plot 3 — Feature Importance...")
fig, ax = plt.subplots(figsize=(8, 6))
colors = sns.color_palette("Blues_d", len(fi_df))[::-1]
bars = ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1],
               color=colors, edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, fi_df["importance"][::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val*100:.2f}%", va="center", fontsize=8)
ax.set_title("Feature Importance — XGBoost")
ax.set_xlabel("Importance Score")
ax.set_ylabel("Feature")
ax.set_xlim(0, fi_df["importance"].max() * 1.18)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/feature_importance.png")

# ── Plot 4: Prediction Distribution ──────────────────────────────────────────
print("  Generating Plot 4 — Prediction Distribution...")
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(actual,    bins=60, alpha=0.55, color="#1976D2",
        label="Actual",    edgecolor="white")
ax.hist(predicted, bins=60, alpha=0.55, color="#E53935",
        label="Predicted", edgecolor="white")
ax.axvline(actual.mean(),    color="#0D47A1", linestyle="--", linewidth=1.5,
           label=f"Actual mean ({actual.mean():.1f})")
ax.axvline(predicted.mean(), color="#B71C1C", linestyle="--", linewidth=1.5,
           label=f"Predicted mean ({predicted.mean():.1f})")
ax.set_title("Distribution of Actual vs Predicted Waste  [XGBoost]")
ax.set_xlabel("Waste Amount (tons)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"), bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/prediction_distribution.png")

# ── Plot 5: RF vs XGBoost Metric Comparison ───────────────────────────────────
# Load RF metrics if available for a side-by-side comparison
rf_metrics_path = os.path.join("Random_Forest_Outputs", "evaluation_metrics.json")
if os.path.exists(rf_metrics_path):
    print("  Generating Plot 5 — RF vs XGBoost Comparison...")
    with open(rf_metrics_path) as f:
        rf_data = json.load(f)
    rf_m = rf_data["metrics"]

    metrics_labels = ["RMSE", "MAE", "R²", "MAPE (%)"]
    rf_vals  = [rf_m["rmse"], rf_m["mae"], rf_m["r2"], rf_m["mape"]]
    xgb_vals = [round(rmse,4), round(mae,4), round(r2,4), round(mape,4)]

    x = np.arange(len(metrics_labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, rf_vals,  width, label="Random Forest",
                color="#1976D2", edgecolor="white")
    b2 = ax.bar(x + width/2, xgb_vals, width, label="XGBoost",
                color="#E53935", edgecolor="white")
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=8)
    ax.set_title("Random Forest vs XGBoost — Metric Comparison")
    ax.set_xticks(x); ax.set_xticklabels(metrics_labels)
    ax.set_ylabel("Score")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rf_vs_xgboost_comparison.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {OUTPUT_DIR}/rf_vs_xgboost_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 11 — Final Output Summary
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
print(pred_df[["actual", "predicted"]].head(10).round(2).to_string(index=False))

# Compare with RF if available
if os.path.exists(rf_metrics_path):
    print(f"\n  {'─'*55}")
    print(f"  MODEL COMPARISON  (Random Forest vs XGBoost)")
    print(f"  {'─'*55}")
    print(f"  {'Metric':<10} {'Random Forest':>15} {'XGBoost':>12} {'Winner':>10}")
    print(f"  {'─'*10} {'─'*15} {'─'*12} {'─'*10}")
    for metric, rf_val, xgb_val in zip(
        ["RMSE", "MAE", "R²", "MAPE"],
        [rf_m["rmse"], rf_m["mae"], rf_m["r2"], rf_m["mape"]],
        [round(rmse,4), round(mae,4), round(r2,4), round(mape,4)]
    ):
        # Lower is better for RMSE/MAE/MAPE; higher is better for R²
        if metric == "R²":
            winner = "XGBoost" if xgb_val > rf_val else "Random Forest"
        else:
            winner = "XGBoost" if xgb_val < rf_val else "Random Forest"
        print(f"  {metric:<10} {rf_val:>15.4f} {xgb_val:>12.4f} {winner:>10}")

print(f"\n  {'─'*55}")
print(f"  FILES SAVED IN  ->  {OUTPUT_DIR}/")
print(f"  {'─'*55}")
outputs = [
    "evaluation_metrics.txt",
    "evaluation_metrics.json",
    "predictions.csv",
    "feature_importance.csv",
    "actual_vs_predicted.png",
    "residual_plot.png",
    "feature_importance.png",
    "prediction_distribution.png",
    "rf_vs_xgboost_comparison.png",
]
for fname in outputs:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        unit = "MB" if size >= 1_048_576 else ("KB" if size >= 1024 else "B")
        size_disp = (size // 1_048_576 if unit == "MB"
                     else size // 1024 if unit == "KB" else size)
        print(f"  ✓  {fname:<42} ({size_disp} {unit})")
    else:
        print(f"  -  {fname:<42} (skipped — RF outputs not found)")

print(f"\n{'█'*60}")
print(f"  XGBoost pipeline complete.  Runtime: {time.time()-t0:.1f}s total")
print(f"{'█'*60}")
