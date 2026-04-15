"""
eda_analysis.py
===============
Exploratory Data Analysis for Smart City Waste Prediction
Dataset: ml_ready_dataset.csv

Produces:
  1. histograms.png         — distribution of 5 numerical features
  2. heatmap.png            — correlation matrix of all ML features
  3. temp_vs_waste.png      — scatter: temperature vs waste_amount
  4. sector_vs_waste.png    — box plot: sector_enc vs waste_amount
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FILE = "ml_ready_dataset.csv"

# Columns for histograms
HIST_COLS = [
    "temperature",
    "humidity",
    "pollution_index",
    "population_density",
    "waste_amount",
]

# Columns for correlation heatmap
CORR_COLS = [
    "temperature", "humidity", "pollution_index", "population_density",
    "sector_enc", "zone_enc", "location_enc",
    "weather_condition_enc", "season_enc",
    "year", "month", "day_of_week", "is_weekend",
    "waste_amount",
]

# Consistent plot style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "axes.titlesize":   12,
    "axes.titleweight": "bold",
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})
sns.set_theme(style="whitegrid", palette="muted")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Dataset
# ─────────────────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)
print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"  Nulls  : {df.isnull().sum().sum()}")
print(f"\nBasic statistics:")
print(df[HIST_COLS].describe().round(2).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Histograms  (matplotlib, 50 bins, 2×3 grid)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating histograms...")

# Human-readable titles and axis labels for each column
HIST_META = {
    "temperature":        ("Temperature Distribution",        "Temperature (°C)",       "#1976D2"),
    "humidity":           ("Humidity Distribution",           "Humidity (%)",            "#43A047"),
    "pollution_index":    ("Pollution Index Distribution",    "AQI Value",               "#E53935"),
    "population_density": ("Population Density Distribution", "People per km²",          "#FB8C00"),
    "waste_amount":       ("Waste Amount Distribution",       "Waste (tons/day)",        "#8E24AA"),
}

n_cols = 3
n_rows = 2   # 5 plots → 2 rows × 3 cols (last cell left empty)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 9))
fig.suptitle("Feature Distributions — Smart City Waste Prediction Dataset",
             fontsize=15, fontweight="bold", y=1.01)

for idx, col in enumerate(HIST_COLS):
    row, col_pos = divmod(idx, n_cols)
    ax = axes[row][col_pos]
    title, xlabel, color = HIST_META[col]

    ax.hist(df[col].dropna(), bins=50, color=color,
            edgecolor="white", linewidth=0.4, alpha=0.85)

    # Overlay mean and median lines
    mean_val   = df[col].mean()
    median_val = df[col].median()
    ax.axvline(mean_val,   color="black",  linestyle="--", linewidth=1.2,
               label=f"Mean {mean_val:.1f}")
    ax.axvline(median_val, color="orange", linestyle=":",  linewidth=1.2,
               label=f"Median {median_val:.1f}")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Hide the unused 6th subplot
axes[1][2].set_visible(False)

plt.tight_layout()
plt.savefig("histograms.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> histograms.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Correlation Heatmap  (seaborn, no annotations)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating correlation heatmap...")

corr_matrix = df[CORR_COLS].corr()

# Mask the upper triangle to keep the heatmap clean
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="coolwarm",
    center=0,
    vmin=-1, vmax=1,
    linewidths=0.4,
    linecolor="white",
    annot=False,          # no annotations — keep it clean
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.75, "label": "Pearson Correlation"},
)
ax.set_title("Feature Correlation Matrix — ML-Ready Dataset",
             fontsize=14, fontweight="bold", pad=14)
plt.xticks(rotation=40, ha="right", fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Scatter Plot: Temperature vs Waste Amount
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating scatter plot: temperature vs waste_amount...")

# Sample 10,000 points so the plot renders quickly without losing pattern
sample = df.sample(n=min(10_000, len(df)), random_state=42)

fig, ax = plt.subplots(figsize=(9, 6))
sc = ax.scatter(
    sample["temperature"], sample["waste_amount"],
    alpha=0.25, s=8,
    c=sample["waste_amount"], cmap="YlOrRd",
    edgecolors="none",
)
plt.colorbar(sc, ax=ax, label="Waste Amount (tons/day)")

# Trend line
m, b = np.polyfit(sample["temperature"], sample["waste_amount"], 1)
x_range = np.linspace(sample["temperature"].min(),
                      sample["temperature"].max(), 200)
ax.plot(x_range, m * x_range + b,
        color="#1565C0", linewidth=2,
        label=f"Trend  y = {m:.2f}x + {b:.1f}")

# Correlation annotation
corr_val = df["temperature"].corr(df["waste_amount"])
ax.text(0.04, 0.93, f"Pearson r = {corr_val:+.4f}",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="lightyellow", alpha=0.85))

ax.set_title("Temperature vs Waste Amount", fontsize=13, fontweight="bold")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Waste Amount (tons/day)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("temp_vs_waste.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> temp_vs_waste.png")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Box Plot: Sector Encoding vs Waste Amount
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating box plot: sector_enc vs waste_amount...")

# Map encoded sector values back to labels for readability
SECTOR_LABELS = {0: "Commercial", 1: "Healthcare",
                 2: "Industrial",  3: "Residential"}
df["sector_label"] = df["sector_enc"].map(SECTOR_LABELS)

# Sector order: Industrial → Commercial → Residential → Healthcare
sector_order = ["Industrial", "Commercial", "Residential", "Healthcare"]
palette      = {"Industrial":  "#E53935",
                "Commercial":  "#1976D2",
                "Residential": "#43A047",
                "Healthcare":  "#FB8C00"}

fig, ax = plt.subplots(figsize=(9, 6))
sns.boxplot(
    data=df,
    x="sector_label",
    y="waste_amount",
    hue="sector_label",
    order=sector_order,
    palette=palette,
    width=0.5,
    linewidth=1.2,
    flierprops=dict(marker="o", markersize=2, alpha=0.3),
    legend=False,
    ax=ax,
)

# Overlay mean markers
means = df.groupby("sector_label")["waste_amount"].mean()
for i, sector in enumerate(sector_order):
    ax.plot(i, means[sector], marker="D", color="white",
            markersize=7, zorder=5, label="Mean" if i == 0 else "")

ax.set_title("Waste Amount Distribution by Sector",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Sector")
ax.set_ylabel("Waste Amount (tons/day)")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("sector_vs_waste.png", dpi=150, bbox_inches="tight")
plt.show()
print("  Saved -> sector_vs_waste.png")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*50)
print("  EDA COMPLETE — Files saved:")
print("="*50)
outputs = [
    "histograms.png",
    "heatmap.png",
    "temp_vs_waste.png",
    "sector_vs_waste.png",
]
import os
for fname in outputs:
    size_kb = os.path.getsize(fname) // 1024
    print(f"  ✓  {fname:<30} ({size_kb} KB)")
print("="*50)
