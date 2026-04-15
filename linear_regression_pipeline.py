"""
linear_regression_pipeline.py  —  STANDALONE SCRIPT
=====================================================
Smart City Waste Prediction — Linear Regression Pipeline

Mirrors xgboost_pipeline.py and random_forest_pipeline.py exactly:
  - Same dataset, same features, same target
  - Same train/test split (80/20, seed=42)
  - Same evaluation metrics (RMSE, MAE, R², MAPE)
  - Same output folder structure
  - Same visualisation plots
  - Plugs directly into the 3-model comparison table

Run: python linear_regression_pipeline.py
"""

import os
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection  import train_test_split
from sklearn.linear_model     import LinearRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.metrics          import (mean_squared_error,
                                      mean_absolute_error,
                                      r2_score)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS  (identical to RF and XGBoost pipelines)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "ml_ready_dataset.csv"
OUTPUT_DIR = "LinearRegression_Outputs"
SEED       = 42          # kept for train_test_split reproducibility
TEST_SIZE  = 0.20

FEATURE_COLS = [
    "temperature", "humidity", "population", "pollution_index",
    "population_density", "year", "month", "day_of_week", "is_weekend",
    "sector_enc", "weather_condition_enc", "location_enc",
    "season_enc", "zone_enc",
]
TARGET_COL = "waste_amount"

# ── Linear Regression has no hyperparameters to tune, but we expose the
#    fit_intercept flag here for easy experimentation.
LR_PARAMS = {
    "fit_intercept": True,   # include bias term (recommended)
    "n_jobs":        -1,     # use all CPU cores for fitting
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Plot style (identical to other pipelines) ─────────────────────────────────
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
    """Print a formatted step header — same style as other pipelines."""
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

# Drop nulls and duplicates (mirrors RF / XGBoost pipelines)
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
for i, col in enumerate(FEATURE_COLS, 1):
    print(f"    {i:>2}. {col}")
print(f"\n  Target column : {TARGET_COL}")
print(f"  X shape       : {X.shape}")
print(f"  y shape       : {y.shape}")
print(f"\n  Target stats  :")
print(y.describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Train / Test Split  (80 / 20 — identical to RF and XGBoost)
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
# STEP 5 — Feature Scaling + Train Linear Regression Model
#
# Linear Regression is sensitive to feature scale — features like
# 'population' (millions) vs 'is_weekend' (0/1) would dominate without
# standardisation. We apply StandardScaler on the training set and
# transform the test set using the same fitted scaler.
# ─────────────────────────────────────────────────────────────────────────────
banner(5, "Feature Scaling + Train LinearRegression")

print("  Model parameters:")
for k, v in LR_PARAMS.items():
    print(f"    {k:<22} = {v}")

# Fit scaler on training data only (prevent data leakage)
scaler  = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n  Feature scaling : StandardScaler applied (zero mean, unit variance)")
print(f"  Scale fitted on : {len(X_train_scaled):,} training rows")

model = LinearRegression(**LR_PARAMS)

print("\n  Training model — please wait...")
t0 = time.time()
model.fit(X_train_scaled, y_train)
elapsed = time.time() - t0
print(f"  Training complete in {elapsed:.3f}s")
print(f"  Intercept       : {model.intercept_:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Generate Predictions
# ─────────────────────────────────────────────────────────────────────────────
banner(6, "Generate Predictions on Test Set")

y_pred = model.predict(X_test_scaled)

# Build predictions DataFrame (same structure as RF / XGBoost)
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
banner(8, f"Save Reports  ->  {OUTPUT_DIR}/")

# 8a. evaluation_metrics.txt
metrics_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.txt")
with open(metrics_path, "w", encoding="utf-8") as f:
    f.write("=" * 55 + "\n")
    f.write("  LINEAR REGRESSION — EVALUATION METRICS\n")
    f.write("=" * 55 + "\n\n")
    f.write(f"  Model          : LinearRegression\n")
    f.write(f"  Dataset        : {INPUT_FILE}\n")
    f.write(f"  Total rows     : {len(df):,}\n")
    f.write(f"  Training rows  : {len(X_train):,}\n")
    f.write(f"  Testing  rows  : {len(X_test):,}\n")
    f.write(f"  fit_intercept  : {LR_PARAMS['fit_intercept']}\n")
    f.write(f"  Scaling        : StandardScaler\n")
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

# 8b. evaluation_metrics.json  (plugs into 3-model comparison)
json_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.json")
metrics_dict = {
    "model":      "LinearRegression",
    "dataset":    INPUT_FILE,
    "total_rows": len(df),
    "train_rows": len(X_train),
    "test_rows":  len(X_test),
    "hyperparameters": {**LR_PARAMS, "scaler": "StandardScaler"},
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
# STEP 9 — Coefficient Analysis  (Linear Regression's "feature importance")
#
# For Linear Regression, the standardised coefficients (coef_ on scaled data)
# represent the change in waste_amount per 1-std-dev change in each feature.
# Larger absolute value = stronger linear influence.
# ─────────────────────────────────────────────────────────────────────────────
banner(9, "Coefficient Analysis  (Feature Influence)")

coef_df = (pd.DataFrame({
               "feature":     FEATURE_COLS,
               "coefficient": model.coef_,
               "abs_coef":    np.abs(model.coef_),
           })
           .sort_values("abs_coef", ascending=False)
           .reset_index(drop=True))
coef_df["rank"] = coef_df.index + 1

# Normalise abs coefficients to 0-1 for a comparable "importance" score
coef_df["importance"]     = coef_df["abs_coef"] / coef_df["abs_coef"].sum()
coef_df["importance_pct"] = (coef_df["importance"] * 100).round(3)

print(f"\n  {'Rank':<5} {'Feature':<25} {'Coefficient':>12} {'|Coef|':>8} {'%':>8}")
print(f"  {'─'*5} {'─'*25} {'─'*12} {'─'*8} {'─'*8}")
for _, row in coef_df.iterrows():
    direction = "+" if row["coefficient"] >= 0 else "-"
    bar = "█" * int(row["importance"] * 50)
    print(f"  {int(row['rank']):<5} {row['feature']:<25} "
          f"{row['coefficient']:>+12.4f} {row['abs_coef']:>8.4f} "
          f"{row['importance_pct']:>7.2f}%  {bar}")

print(f"\n  Intercept : {model.intercept_:.4f}")

# Save coefficients as feature_importance.csv (same filename as other pipelines)
fi_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
coef_df[["rank", "feature", "coefficient", "abs_coef",
         "importance", "importance_pct"]].to_csv(fi_csv, index=False)
print(f"\n  Saved -> {fi_csv}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 10 — Visualisations  (same 4 plots as RF / XGBoost + comparison)
# ─────────────────────────────────────────────────────────────────────────────
banner(10, "Generate Visualisations")

actual    = pred_df["actual"].values
predicted = pred_df["predicted"].values
residuals = actual - predicted

# Sample for scatter plots (keep rendering fast)
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
ax.set_title("Actual vs Predicted Waste Amount  [Linear Regression]")
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
ax.axhline(0,     color="red",    linewidth=1.8, linestyle="--",
           label="Zero residual")
ax.axhline( rmse, color="orange", linewidth=1,   linestyle=":", alpha=0.8,
            label=f"+RMSE ({rmse:.1f})")
ax.axhline(-rmse, color="orange", linewidth=1,   linestyle=":", alpha=0.8,
            label=f"-RMSE ({rmse:.1f})")
ax.set_title("Residual Plot  (Actual - Predicted)  [Linear Regression]")
ax.set_xlabel("Predicted Waste (tons)")
ax.set_ylabel("Residual (tons)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "residual_plot.png"), bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/residual_plot.png")

# ── Plot 3: Coefficient Bar Chart  (equivalent of feature importance) ─────────
print("  Generating Plot 3 — Coefficient / Feature Influence...")
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#E53935" if c < 0 else "#1976D2"
          for c in coef_df["coefficient"][::-1]]
bars = ax.barh(coef_df["feature"][::-1], coef_df["coefficient"][::-1],
               color=colors, edgecolor="white", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
for bar, val in zip(bars, coef_df["coefficient"][::-1]):
    xpos = bar.get_width() + (0.5 if val >= 0 else -0.5)
    ha   = "left" if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}", va="center", fontsize=7.5, ha=ha)
ax.set_title("Standardised Coefficients — Linear Regression\n"
             "(Blue = positive effect, Red = negative effect)")
ax.set_xlabel("Coefficient Value  (on standardised features)")
ax.set_ylabel("Feature")
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
ax.set_title("Distribution of Actual vs Predicted Waste  [Linear Regression]")
ax.set_xlabel("Waste Amount (tons)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=9)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "prediction_distribution.png"),
            bbox_inches="tight")
plt.close()
print(f"  Saved -> {OUTPUT_DIR}/prediction_distribution.png")

# ── Plot 5: 3-Model Comparison  (RF vs XGBoost vs Linear Regression) ──────────
rf_path  = os.path.join("Random_Forest_Outputs",  "evaluation_metrics.json")
xgb_path = os.path.join("XGBoost_Outputs",        "evaluation_metrics.json")

if os.path.exists(rf_path) and os.path.exists(xgb_path):
    print("  Generating Plot 5 — 3-Model Comparison...")
    with open(rf_path)  as f: rf_m  = json.load(f)["metrics"]
    with open(xgb_path) as f: xgb_m = json.load(f)["metrics"]
    lr_m = {"rmse": round(rmse,4), "mae": round(mae,4),
            "r2":   round(r2,4),   "mape": round(mape,4)}

    metric_labels = ["RMSE", "MAE", "R²", "MAPE (%)"]
    rf_vals  = [rf_m["rmse"],  rf_m["mae"],  rf_m["r2"],  rf_m["mape"]]
    xgb_vals = [xgb_m["rmse"], xgb_m["mae"], xgb_m["r2"], xgb_m["mape"]]
    lr_vals  = [lr_m["rmse"],  lr_m["mae"],  lr_m["r2"],  lr_m["mape"]]

    x     = np.arange(len(metric_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width,     rf_vals,  width, label="Random Forest",
                color="#1976D2", edgecolor="white")
    b2 = ax.bar(x,             xgb_vals, width, label="XGBoost",
                color="#E53935", edgecolor="white")
    b3 = ax.bar(x + width,     lr_vals,  width, label="Linear Regression",
                color="#43A047", edgecolor="white")
    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7.5)
    ax.set_title("Model Comparison — Random Forest vs XGBoost vs Linear Regression")
    ax.set_xticks(x); ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "3model_comparison.png"),
                bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {OUTPUT_DIR}/3model_comparison.png")

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
print(f"  TOP 5 FEATURE INFLUENCES  (by |coefficient|)")
print(f"  {'─'*55}")
for _, row in coef_df.head(5).iterrows():
    sign = "+" if row["coefficient"] >= 0 else "-"
    print(f"  {int(row['rank'])}. {row['feature']:<25} "
          f"{row['coefficient']:>+10.4f}  ({row['importance_pct']:.2f}%)")

print(f"\n  {'─'*55}")
print(f"  SAMPLE PREDICTIONS (first 10 rows)")
print(f"  {'─'*55}")
print(pred_df[["actual", "predicted"]].head(10).round(2).to_string(index=False))

# ── 3-model comparison table ──────────────────────────────────────────────────
if os.path.exists(rf_path) and os.path.exists(xgb_path):
    print(f"\n  {'─'*60}")
    print(f"  3-MODEL COMPARISON  (RF vs XGBoost vs Linear Regression)")
    print(f"  {'─'*60}")
    print(f"  {'Metric':<8} {'Rand Forest':>12} {'XGBoost':>10} "
          f"{'Linear Reg':>12} {'Best':>14}")
    print(f"  {'─'*8} {'─'*12} {'─'*10} {'─'*12} {'─'*14}")

    for metric, rf_val, xgb_val, lr_val in zip(
        ["RMSE", "MAE", "R²", "MAPE"],
        [rf_m["rmse"],  rf_m["mae"],  rf_m["r2"],  rf_m["mape"]],
        [xgb_m["rmse"], xgb_m["mae"], xgb_m["r2"], xgb_m["mape"]],
        [round(rmse,4), round(mae,4), round(r2,4),  round(mape,4)],
    ):
        vals = {"Random Forest": rf_val, "XGBoost": xgb_val,
                "Linear Reg":   lr_val}
        # R² → higher is better; all others → lower is better
        best = max(vals, key=vals.get) if metric == "R²" \
               else min(vals, key=vals.get)
        print(f"  {metric:<8} {rf_val:>12.4f} {xgb_val:>10.4f} "
              f"{lr_val:>12.4f} {best:>14}")

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
    "3model_comparison.png",
]
for fname in outputs:
    fpath = os.path.join(OUTPUT_DIR, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        unit = "MB" if size >= 1_048_576 else ("KB" if size >= 1024 else "B")
        size_disp = (size // 1_048_576 if unit == "MB"
                     else size // 1024 if unit == "KB" else size)
        print(f"  ✓  {fname:<44} ({size_disp} {unit})")
    else:
        print(f"  -  {fname:<44} (skipped — other model outputs not found)")

print(f"\n{'█'*60}")
print(f"  Linear Regression pipeline complete.  "
      f"Runtime: {time.time()-t0:.1f}s total")
print(f"{'█'*60}")
