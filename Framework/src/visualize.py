"""
visualize.py
------------
Visualization functions for the Smart City Waste Prediction Framework.
All functions return matplotlib Figure objects for st.pyplot() embedding.
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Consistent global style ───────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "grid.linestyle":   "--",
    "axes.titlesize":   13,
    "axes.titleweight": "bold",
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})


def plot_future_trend(
    df: pd.DataFrame,
    city: str,
    sector: str,
    df2: pd.DataFrame = None,
    city2: str = None,
    sector2: str = None,
) -> plt.Figure:
    """
    Enhanced line plot — predicted waste over future days.
    Supports optional second series for comparison mode.

    Parameters
    ----------
    df      : Primary prediction DataFrame
    city    : Primary city name
    sector  : Primary sector name
    df2     : (optional) Second prediction DataFrame for comparison
    city2   : (optional) Second city name
    sector2 : (optional) Second sector name

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # ── Primary series ────────────────────────────────────────────────────────
    ax.plot(
        df["date"], df["predicted_waste"],
        marker="o", linewidth=2.4, markersize=7,
        color="#1976D2", zorder=3,
        label=f"{city} / {sector}",
    )
    ax.fill_between(df["date"], df["predicted_waste"],
                    alpha=0.08, color="#1976D2")

    # Highlight peak point with a larger scatter marker
    max_idx = df["predicted_waste"].idxmax()
    min_idx = df["predicted_waste"].idxmin()
    ax.scatter(df.loc[max_idx, "date"], df.loc[max_idx, "predicted_waste"],
               s=120, color="#E53935", zorder=5, label="Peak")
    ax.scatter(df.loc[min_idx, "date"], df.loc[min_idx, "predicted_waste"],
               s=120, color="#43A047", zorder=5, label="Lowest")

    # Annotate peak and lowest
    ax.annotate(
        f"Peak\n{df.loc[max_idx,'predicted_waste']:.0f} t",
        xy=(df.loc[max_idx, "date"], df.loc[max_idx, "predicted_waste"]),
        xytext=(0, 16), textcoords="offset points",
        ha="center", fontsize=8, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
    )
    ax.annotate(
        f"Min\n{df.loc[min_idx,'predicted_waste']:.0f} t",
        xy=(df.loc[min_idx, "date"], df.loc[min_idx, "predicted_waste"]),
        xytext=(0, -24), textcoords="offset points",
        ha="center", fontsize=8, color="#2E7D32",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2),
    )

    # ── Optional second series (comparison mode) ──────────────────────────────
    if df2 is not None and city2 is not None:
        ax.plot(
            df2["date"], df2["predicted_waste"],
            marker="s", linewidth=2.4, markersize=7,
            color="#E53935", zorder=3, linestyle="--",
            label=f"{city2} / {sector2}",
        )
        ax.fill_between(df2["date"], df2["predicted_waste"],
                        alpha=0.06, color="#E53935")

    ax.set_title(f"Future Waste Forecast — {city} / {sector}"
                 + (f"  vs  {city2} / {sector2}" if city2 else ""))
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Waste (tons/day)")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.35, linestyle="--")
    plt.tight_layout()
    return fig


def plot_prediction_distribution(df: pd.DataFrame, city: str) -> plt.Figure:
    """
    Histogram of predicted waste values across the forecast period.

    Returns matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    bins = min(20, max(5, len(df) // 2))
    ax.hist(df["predicted_waste"], bins=bins,
            color="#1976D2", edgecolor="white", alpha=0.85)

    mean_val = df["predicted_waste"].mean()
    ax.axvline(mean_val, color="#E53935", linestyle="--", linewidth=1.8,
               label=f"Mean: {mean_val:.1f} tons")

    ax.set_title(f"Predicted Waste Distribution — {city}")
    ax.set_xlabel("Predicted Waste (tons/day)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}")
    )
    plt.tight_layout()
    return fig


def plot_heatmap(df: pd.DataFrame) -> plt.Figure:
    """
    Feature impact heatmap — shows correlation between features and
    predicted waste. Handles constant columns (zero variance) gracefully
    by replacing NaN correlations with 0.

    Title: "Feature Impact on Predicted Waste"
    Returns matplotlib Figure.
    """
    wanted = [
        "predicted_waste", "temperature", "humidity",
        "pollution_index", "population_density",
    ]
    cols = [c for c in wanted if c in df.columns]

    # Drop constant columns (zero std → NaN correlation) before computing
    varying = [c for c in cols if df[c].std() > 0]
    # Always keep predicted_waste even if somehow constant
    if "predicted_waste" not in varying and "predicted_waste" in cols:
        varying = ["predicted_waste"] + varying

    corr = df[varying].corr().fillna(0)

    # Friendly display labels
    label_map = {
        "predicted_waste":    "Predicted\nWaste",
        "temperature":        "Temperature",
        "humidity":           "Humidity",
        "pollution_index":    "Pollution\nIndex",
        "population_density": "Pop.\nDensity",
    }
    corr.index   = [label_map.get(c, c) for c in corr.index]
    corr.columns = [label_map.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        corr,
        annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        linewidths=1.2, linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
        square=True,
    )
    ax.set_title("Feature Impact on Predicted Waste",
                 fontsize=13, fontweight="bold", pad=14)
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig
