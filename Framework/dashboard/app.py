"""
app.py  —  Smart City Waste Prediction Dashboard  (Upgraded)
=============================================================
Features:
  • KPI cards with delta labels
  • Alert system (error / warning / success)
  • Enhanced trend graph with peak highlight
  • Improved heatmap with 5 key features
  • Comparison mode (two cities on one chart)
  • CSV + JSON download
  • City map visualization (st.map)
  • Project info panel
  • Clean modular code

Run:
    cd Framework
    python -m streamlit run dashboard/app.py
"""

import sys, os

# ── Resolve src/ path regardless of working directory ────────────────────────
_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
sys.path.insert(0, _SRC)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

from predict   import predict_future
from visualize import (plot_future_trend,
                       plot_prediction_distribution,
                       plot_heatmap)
from utils     import CITY_MAP, SECTOR_MAP, ZONE_MAP

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── City coordinates for map visualization ────────────────────────────────────
CITY_COORDS = {
    "Ahmedabad":     [23.02, 72.57],
    "Bangalore":     [12.97, 77.59],
    "Bhopal":        [23.26, 77.41],
    "Chennai":       [13.08, 80.27],
    "Coimbatore":    [11.00, 76.96],
    "Delhi":         [28.61, 77.20],
    "Hyderabad":     [17.38, 78.47],
    "Indore":        [22.72, 75.86],
    "Jaipur":        [26.91, 75.79],
    "Kochi":         [ 9.93, 76.26],
    "Kolkata":       [22.57, 88.36],
    "Lucknow":       [26.85, 80.95],
    "Mumbai":        [19.07, 72.87],
    "Nagpur":        [21.14, 79.08],
    "Patna":         [25.59, 85.13],
    "Pune":          [18.52, 73.85],
    "Surat":         [21.17, 72.83],
    "Visakhapatnam": [17.69, 83.22],
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute summary KPI values from a prediction DataFrame."""
    return {
        "avg":   round(df["predicted_waste"].mean(), 1),
        "peak":  round(df["predicted_waste"].max(),  1),
        "low":   round(df["predicted_waste"].min(),  1),
        "total": round(df["predicted_waste"].sum(),  0),
    }


def show_kpi_cards(kpis: dict, label: str = "") -> None:
    """Render 4 KPI metric cards in a single row."""
    prefix = f"{label} — " if label else ""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"📦 {prefix}Avg Daily Waste",
              f"{kpis['avg']:.1f} tons",
              delta=None)
    c2.metric(f"📈 {prefix}Peak Day",
              f"{kpis['peak']:.1f} tons",
              delta="High",
              delta_color="inverse")
    c3.metric(f"📉 {prefix}Lowest Day",
              f"{kpis['low']:.1f} tons",
              delta="Low",
              delta_color="normal")
    c4.metric(f"🗑️ {prefix}Total Forecast",
              f"{kpis['total']:,.0f} tons",
              delta=None)


def show_alert(peak: float) -> None:
    """Display a colour-coded alert based on peak predicted waste."""
    if peak > 800:
        st.error(
            f"🚨 **High Waste Alert** — Peak predicted waste is **{peak:.1f} tons/day**. "
            "Immediate waste management action recommended."
        )
    elif peak > 650:
        st.warning(
            f"⚠️ **Moderate Waste Level** — Peak predicted waste is **{peak:.1f} tons/day**. "
            "Monitor closely and prepare additional capacity."
        )
    else:
        st.success(
            f"✅ **Waste Under Control** — Peak predicted waste is **{peak:.1f} tons/day**. "
            "Current waste management capacity is sufficient."
        )


def show_city_map(
    city: str,
    avg_waste: float,
    city2: str = None,
    avg_waste2: float = None,
) -> None:
    """
    Render an enhanced city map using st.map().

    Shows ALL 18 cities as background context, with the selected
    city/cities highlighted via a separate info block.
    Dot size is proportional to predicted waste level.

    Parameters
    ----------
    city       : Primary selected city
    avg_waste  : Primary city average predicted waste (for display)
    city2      : (optional) Second city in comparison mode
    avg_waste2 : (optional) Second city average predicted waste
    """
    # Build full map DataFrame — all 18 cities shown as context
    all_rows = []
    for cname, (lat, lon) in CITY_COORDS.items():
        all_rows.append({"lat": lat, "lon": lon, "city": cname})
    map_df = pd.DataFrame(all_rows)

    # Show the map (all 18 cities)
    st.map(map_df[["lat", "lon"]], zoom=4)

    # City info cards below the map
    selected = [(city, avg_waste)]
    if city2 and avg_waste2 is not None:
        selected.append((city2, avg_waste2))

    info_cols = st.columns(len(selected))
    for col, (cname, waste) in zip(info_cols, selected):
        lat, lon = CITY_COORDS.get(cname, [0, 0])
        waste_level = (
            "🔴 High"   if waste > 800 else
            "🟡 Moderate" if waste > 650 else
            "🟢 Normal"
        )
        with col:
            st.markdown(
                f"""
                <div style="background:#E3F2FD; border-radius:10px;
                            padding:0.8rem 1rem; border-left:4px solid #1976D2;">
                    <b>📍 {cname}</b><br>
                    🌐 {lat:.2f}°N, {lon:.2f}°E<br>
                    🗑️ Avg Waste: <b>{waste:.1f} tons/day</b><br>
                    📊 Level: {waste_level}
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_model_plots() -> None:
    """Display the 4 pre-generated XGBoost evaluation plots."""
    plot_files = {
        "Actual vs Predicted":  "actual_vs_predicted.png",
        "Residual Plot":        "residual_plot.png",
        "Feature Importance":   "feature_importance.png",
    }
    img_cols = st.columns(len(plot_files))
    for col, (title, fname) in zip(img_cols, plot_files.items()):
        fpath = os.path.join(OUTPUTS_DIR, fname)
        with col:
            st.markdown(f"**{title}**")
            if os.path.exists(fpath):
                st.image(Image.open(fpath), use_container_width=True)
            else:
                st.warning(f"`outputs/{fname}` not found")

    dist_path = os.path.join(OUTPUTS_DIR, "prediction_distribution.png")
    if os.path.exists(dist_path):
        st.markdown("**Prediction Distribution (Training Data)**")
        st.image(Image.open(dist_path), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart City Waste Prediction",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-title {
        font-size: 2.3rem; font-weight: 800;
        color: #0D47A1; margin-bottom: 0.1rem;
    }
    .sub-title {
        font-size: 1rem; color: #546E7A; margin-bottom: 1rem;
    }
    /* Section headers */
    .sec-header {
        font-size: 1.1rem; font-weight: 700; color: #1565C0;
        border-bottom: 2px solid #BBDEFB;
        padding-bottom: 0.25rem; margin: 1.2rem 0 0.6rem 0;
    }
    /* Welcome cards */
    .info-card {
        background: #E3F2FD; border-radius: 10px;
        padding: 1rem 1.2rem; border-left: 5px solid #1976D2;
        margin-bottom: 0.5rem;
    }
    /* Generate button */
    .stButton > button {
        background: linear-gradient(135deg, #1976D2, #0D47A1);
        color: white; border-radius: 8px; border: none;
        padding: 0.55rem 1.5rem; font-size: 1rem;
        font-weight: 700; width: 100%;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }
    /* Dataframe */
    .stDataFrame { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">♻️ Smart City Waste Prediction Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">'
    'AI-powered daily waste forecasting across 18 Indian cities · '
    'XGBoost Model · R² = 0.9838'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Prediction Settings")
    st.divider()

    # ── Primary inputs ────────────────────────────────────────────────────────
    st.markdown("### 🏙️ Primary City")
    city = st.selectbox(
        "City", options=sorted(CITY_MAP.keys()),
        index=sorted(CITY_MAP.keys()).index("Delhi"),
        label_visibility="collapsed",
    )
    sector = st.selectbox(
        "Sector", options=list(SECTOR_MAP.keys()), index=2,
        label_visibility="collapsed",
    )
    zone = st.selectbox(
        "Zone", options=list(ZONE_MAP.keys()), index=0,
        label_visibility="collapsed",
    )
    days = st.slider("📅 Forecast Days", 1, 14, 7)

    st.divider()

    # ── Comparison mode toggle ────────────────────────────────────────────────
    compare = st.checkbox("⚔️ Enable Comparison Mode",
                          help="Compare two cities side by side")

    city2 = sector2 = zone2 = None
    if compare:
        st.markdown("### 🏙️ Second City")
        city2 = st.selectbox(
            "City 2", options=sorted(CITY_MAP.keys()),
            index=sorted(CITY_MAP.keys()).index("Mumbai"),
            key="city2", label_visibility="collapsed",
        )
        sector2 = st.selectbox(
            "Sector 2", options=list(SECTOR_MAP.keys()), index=2,
            key="sector2", label_visibility="collapsed",
        )
        zone2 = st.selectbox(
            "Zone 2", options=list(ZONE_MAP.keys()), index=1,
            key="zone2", label_visibility="collapsed",
        )

    st.divider()
    st.info(
        f"**Primary:** {city} · {sector} · {zone}\n\n"
        + (f"**Compare:** {city2} · {sector2} · {zone2}\n\n" if compare else "")
        + f"**Days:** {days}"
    )

    generate = st.button("🚀 Generate Prediction", use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# WELCOME SCREEN (before prediction)
# ─────────────────────────────────────────────────────────────────────────────
if not generate:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="info-card">
            <b>🤖 Model</b><br>XGBoost Regressor<br>
            R² = 0.9838 &nbsp;|&nbsp; RMSE = 34.88 tons
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="info-card">
            <b>🏙️ Coverage</b><br>18 Indian Cities<br>
            4 Sectors &nbsp;·&nbsp; 5 Zones
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="info-card">
            <b>📊 Dataset</b><br>263,160 rows<br>
            2 years of data (2023–2024)
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "### 👈 Configure settings in the sidebar and click "
        "**Generate Prediction** to begin."
    )

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
else:
    # ── Run primary prediction ────────────────────────────────────────────────
    with st.spinner(f"Predicting — {city} / {sector} / {zone} ({days} days)…"):
        try:
            result_df = predict_future(city=city, sector=sector,
                                       zone=zone, days=days)
            success = True
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
            success = False

    # ── Run comparison prediction (if enabled) ────────────────────────────────
    result_df2 = None
    if success and compare and city2:
        with st.spinner(f"Predicting — {city2} / {sector2} / {zone2}…"):
            try:
                result_df2 = predict_future(city=city2, sector=sector2,
                                            zone=zone2, days=days)
            except Exception as e:
                st.warning(f"⚠️ Comparison prediction failed: {e}")

    if success:
        kpis = compute_kpis(result_df)

        # ── 1. ALERT ─────────────────────────────────────────────────────────
        show_alert(kpis["peak"])

        st.divider()

        # ── 2. KPI CARDS ─────────────────────────────────────────────────────
        st.markdown('<div class="sec-header">📊 Predictions</div>',
                    unsafe_allow_html=True)
        show_kpi_cards(kpis, label=city)

        # Second city KPIs side by side
        if result_df2 is not None:
            st.markdown("---")
            kpis2 = compute_kpis(result_df2)
            show_kpi_cards(kpis2, label=city2)
        else:
            kpis2 = None

        st.divider()

        # ── 3. PREDICTION TABLE ───────────────────────────────────────────────
        st.markdown('<div class="sec-header">📋 Prediction Table</div>',
                    unsafe_allow_html=True)

        if result_df2 is not None:
            tab1, tab2 = st.tabs([f"📍 {city}", f"📍 {city2}"])
            with tab1:
                disp1 = result_df.copy()
                disp1.columns = [c.replace("_", " ").title()
                                  for c in disp1.columns]
                st.dataframe(disp1, use_container_width=True, height=260)
            with tab2:
                disp2 = result_df2.copy()
                disp2.columns = [c.replace("_", " ").title()
                                  for c in disp2.columns]
                st.dataframe(disp2, use_container_width=True, height=260)
        else:
            disp = result_df.copy()
            disp.columns = [c.replace("_", " ").title() for c in disp.columns]
            st.dataframe(disp, use_container_width=True, height=260)

        # ── 4. DOWNLOAD OPTIONS ───────────────────────────────────────────────
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                label="⬇️ Download CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name=f"waste_{city}_{sector}_{days}d.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl_col2:
            st.download_button(
                label="⬇️ Download JSON",
                data=result_df.to_json(orient="records", indent=2),
                file_name=f"waste_{city}_{sector}_{days}d.json",
                mime="application/json",
                use_container_width=True,
            )

        st.divider()

        # ── 5. TREND ANALYSIS ─────────────────────────────────────────────────
        st.markdown('<div class="sec-header">📈 Trend Analysis</div>',
                    unsafe_allow_html=True)

        fig_trend = plot_future_trend(
            result_df, city, sector,
            df2=result_df2, city2=city2, sector2=sector2,
        )
        st.pyplot(fig_trend, use_container_width=True)

        st.divider()

        # ── 6. DISTRIBUTION + HEATMAP ─────────────────────────────────────────
        st.markdown('<div class="sec-header">🔥 Insights</div>',
                    unsafe_allow_html=True)

        ins_left, ins_right = st.columns([1, 1])
        with ins_left:
            st.markdown("**📊 Prediction Distribution**")
            fig_dist = plot_prediction_distribution(result_df, city)
            st.pyplot(fig_dist, use_container_width=True)

        with ins_right:
            st.markdown("**🔥 Feature Impact Heatmap**")
            st.caption(
                "Pearson correlation between each feature and predicted waste. "
                "Green = positive impact, Red = negative impact."
            )
            fig_heat = plot_heatmap(result_df)
            st.pyplot(fig_heat, use_container_width=True)

        st.divider()

        # ── 7. CITY MAP ───────────────────────────────────────────────────────
        st.markdown('<div class="sec-header">📍 City Map — All 18 Cities</div>',
                    unsafe_allow_html=True)
        st.caption("All 18 Indian cities shown on map. Selected city details displayed below.")
        show_city_map(
            city,
            avg_waste=kpis["avg"],
            city2=city2 if compare else None,
            avg_waste2=kpis2["avg"] if (compare and result_df2 is not None) else None,
        )

        st.divider()

        # ── 8. MODEL EVALUATION PLOTS ─────────────────────────────────────────
        st.markdown('<div class="sec-header">🧠 Model Evaluation Plots (XGBoost)</div>',
                    unsafe_allow_html=True)
        st.caption(
            "Generated during training on 263,160 rows · "
            "80/20 train-test split · seed=42"
        )
        show_model_plots()

        st.divider()

        # ── 9. PROJECT INFO PANEL ─────────────────────────────────────────────
        st.info("""
**Model:** XGBoost Regressor  
**R² Score:** 0.9838  
**RMSE:** 34.88 tons  
**MAE:** 22.99 tons  
**Dataset Size:** 263,160 rows  
**Training Period:** 2023–2024 (2 years)  
**Cities:** 18 major Indian cities  
**Use Case:** Smart City Waste Prediction & Management  
        """)

        st.success(
            f"✅ Forecast complete — **{days} days** for "
            f"**{city}** / **{sector}** / **{zone}**"
            + (f"  vs  **{city2}** / **{sector2}** / **{zone2}**"
               if compare and city2 else "")
        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>"
    "Smart City Waste Prediction System &nbsp;·&nbsp; "
    "XGBoost Model &nbsp;·&nbsp; "
    "18 Indian Cities &nbsp;·&nbsp; "
    "2023–2024 Training Data"
    "</small></center>",
    unsafe_allow_html=True,
)
