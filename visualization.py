"""
visualization.py  —  STANDALONE SCRIPT
=======================================
Run: python visualization.py
Requires: cleaned_dataset.csv  (produced by data_cleaning.py)

Tasks:
  1. Start SparkSession
  2. Load cleaned_dataset.csv
  3. Convert to pandas
  4. Generate 5 plots:
       Plot 1 — Bar chart   : sector vs average waste
       Plot 2 — Line chart  : monthly waste trend
       Plot 3 — Scatter     : temperature vs waste
       Plot 4 — Scatter     : population vs waste (city-level)
       Plot 5 — Heatmap     : feature correlation matrix
  5. Save all plots as PNG files
  6. Display all plots
"""

import os
import tempfile

# ── Windows: stub HADOOP_HOME ─────────────────────────────────────────────────
if os.name == "nt" and not os.environ.get("HADOOP_HOME"):
    _stub = os.path.join(tempfile.gettempdir(), "hadoop_stub")
    os.makedirs(os.path.join(_stub, "bin"), exist_ok=True)
    os.environ["HADOOP_HOME"] = _stub

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pyspark.sql import SparkSession

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi":        120,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.facecolor":  "white",
})

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — SparkSession
# ─────────────────────────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("WasteProject_Visualization")
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
      .csv("cleaned_dataset.csv"))

print(f"Loaded cleaned_dataset.csv — {df.count():,} rows × {len(df.columns)} cols")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Convert to pandas
# ─────────────────────────────────────────────────────────────────────────────
pdf = df.toPandas()
print(f"Converted to pandas: {pdf.shape}")

# Pre-compute aggregates used across plots
sector_avg = (pdf.groupby("sector")["waste_amount"]
                 .mean()
                 .sort_values(ascending=False)
                 .reset_index()
                 .rename(columns={"waste_amount": "avg_waste"}))

month_avg = (pdf.groupby("month")["waste_amount"]
                .mean()
                .reset_index()
                .sort_values("month"))
month_avg["month_name"] = month_avg["month"].map(MONTH_NAMES)

city_avg = (pdf.groupby("location")
               .agg(avg_waste=("waste_amount", "mean"),
                    population=("population", "first"))
               .reset_index())

sample = pdf.sample(n=min(6000, len(pdf)), random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Bar chart: Sector vs Average Waste
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating Plot 1 — Sector vs Average Waste (bar chart)...")

fig, ax = plt.subplots(figsize=(7, 4.5))
palette = sns.color_palette("Set2", len(sector_avg))
bars = ax.bar(sector_avg["sector"], sector_avg["avg_waste"],
              color=palette, edgecolor="white", linewidth=0.8, zorder=3)
ax.bar_label(bars, fmt="%.1f", padding=5, fontsize=9, fontweight="bold")
ax.set_title("Average Daily Waste by Sector")
ax.set_xlabel("Sector")
ax.set_ylabel("Average Waste (tons/day)")
ax.set_ylim(0, sector_avg["avg_waste"].max() * 1.18)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("plot_sector_waste.png", bbox_inches="tight")
print("  Saved → plot_sector_waste.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Line chart: Monthly Waste Trend
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 2 — Monthly Waste Trend (line chart)...")

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(month_avg["month_name"], month_avg["waste_amount"],
        marker="o", linewidth=2.2, markersize=7,
        color="#1976D2", zorder=3, label="Avg Waste")
ax.fill_between(month_avg["month_name"], month_avg["waste_amount"],
                alpha=0.10, color="#1976D2")

# Annotate peak month
peak_idx = month_avg["waste_amount"].idxmax()
peak_row = month_avg.loc[peak_idx]
ax.annotate(f"Peak\n{peak_row['waste_amount']:.0f} t",
            xy=(peak_row["month_name"], peak_row["waste_amount"]),
            xytext=(0, 16), textcoords="offset points",
            ha="center", fontsize=8, color="#C62828",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))

ax.set_title("Monthly Average Waste Trend")
ax.set_xlabel("Month")
ax.set_ylabel("Average Waste (tons/day)")
ax.tick_params(axis="x", rotation=30)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("plot_monthly_trend.png", bbox_inches="tight")
print("  Saved → plot_monthly_trend.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Scatter: Temperature vs Waste
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 3 — Temperature vs Waste (scatter)...")

fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(sample["temperature"], sample["waste_amount"],
                alpha=0.22, s=7,
                c=sample["waste_amount"], cmap="YlOrRd",
                edgecolors="none")
plt.colorbar(sc, ax=ax, label="Waste Amount (tons)")

# Trend line
m, b = np.polyfit(sample["temperature"], sample["waste_amount"], 1)
x_range = np.linspace(sample["temperature"].min(),
                      sample["temperature"].max(), 200)
ax.plot(x_range, m * x_range + b,
        color="#E53935", linewidth=2,
        label=f"Trend  y = {m:.2f}x + {b:.1f}")
ax.legend(fontsize=9)
ax.set_title("Temperature vs Waste Amount")
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Waste Amount (tons)")
plt.tight_layout()
plt.savefig("plot_temp_waste.png", bbox_inches="tight")
print("  Saved → plot_temp_waste.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Scatter: Population vs Waste (city-level)
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 4 — Population vs Waste (scatter)...")

fig, ax = plt.subplots(figsize=(8, 5.5))
sc = ax.scatter(city_avg["population"] / 1e6, city_avg["avg_waste"],
                s=130, c=city_avg["avg_waste"], cmap="Blues",
                edgecolors="#333", linewidths=0.7, zorder=3)
plt.colorbar(sc, ax=ax, label="Avg Waste (tons/day)")

for _, row in city_avg.iterrows():
    ax.annotate(row["location"],
                (row["population"] / 1e6, row["avg_waste"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 3), textcoords="offset points")

# Trend line
m2, b2 = np.polyfit(city_avg["population"] / 1e6, city_avg["avg_waste"], 1)
x2 = np.linspace((city_avg["population"] / 1e6).min(),
                 (city_avg["population"] / 1e6).max(), 100)
ax.plot(x2, m2 * x2 + b2, color="#E53935", linewidth=1.5,
        linestyle="--", label=f"Trend  y = {m2:.1f}x + {b2:.0f}")
ax.legend(fontsize=9)
ax.set_title("City Population vs Average Daily Waste")
ax.set_xlabel("Population (millions)")
ax.set_ylabel("Average Waste (tons/day)")
plt.tight_layout()
plt.savefig("plot_pop_waste.png", bbox_inches="tight")
print("  Saved → plot_pop_waste.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Heatmap: Feature Correlation Matrix
# ─────────────────────────────────────────────────────────────────────────────
print("Generating Plot 5 — Correlation Heatmap...")

corr_cols = [
    "temperature", "humidity", "population", "pollution_index",
    "population_density", "month", "day_of_week", "is_weekend",
    "sector_enc", "season_enc", "zone_enc",
    "weather_condition_enc", "location_enc", "waste_amount",
]
corr_matrix = pdf[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # hide upper triangle

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr_matrix, mask=mask,
            annot=True, fmt=".2f",
            cmap="coolwarm", center=0,
            linewidths=0.4, linecolor="white",
            annot_kws={"size": 7.5}, ax=ax,
            cbar_kws={"shrink": 0.75})
ax.set_title("Feature Correlation Matrix")
plt.xticks(rotation=40, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", bbox_inches="tight")
print("  Saved → plot_correlation_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Display all plots
# ─────────────────────────────────────────────────────────────────────────────
print("\nDisplaying all plots...")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("  VISUALISATION SUMMARY")
print(f"{'='*55}")
plots = [
    "plot_sector_waste.png",
    "plot_monthly_trend.png",
    "plot_temp_waste.png",
    "plot_pop_waste.png",
    "plot_correlation_heatmap.png",
]
for p in plots:
    size_kb = os.path.getsize(p) // 1024
    print(f"  ✓  {p:<35} ({size_kb} KB)")
print(f"{'='*55}")

spark.stop()
print("\nSparkSession stopped. visualization.py complete.")
