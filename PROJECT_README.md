# Smart City Waste Prediction — Complete Project README

## Project Overview

This project builds an end-to-end Machine Learning pipeline to predict daily
waste generation across 18 major Indian cities. Real weather data is fetched
from a public API, expanded into a large synthetic dataset, cleaned and
feature-engineered, analysed through EDA, and finally used to train and
compare three regression models: Random Forest, XGBoost, and Linear Regression.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Environment Setup](#2-environment-setup)
3. [Data Generation](#3-data-generation)
4. [Data Preprocessing & Cleaning](#4-data-preprocessing--cleaning)
5. [Feature Selection](#5-feature-selection)
6. [Exploratory Data Analysis (EDA)](#6-exploratory-data-analysis-eda)
7. [Machine Learning — Model Training](#7-machine-learning--model-training)
8. [Prediction Outputs](#8-prediction-outputs)
9. [Visualizations](#9-visualizations)
10. [Model Comparison Results](#10-model-comparison-results)
11. [How to Run — Step by Step](#11-how-to-run--step-by-step)

---

## 1. Project Structure

```
Smart_City_Waste_Prediction/
│
├── generate_waste_dataset.py       # Step 1 — Fetch API data + generate dataset
├── data_cleaning.py                # Step 2 — Clean raw dataset
├── feature_selection.py            # Step 3 — Select ML-ready features
├── eda_analysis.py                 # Step 4 — Exploratory Data Analysis
├── visualization.py                # Step 5 — EDA visualizations (Spark-based)
├── random_forest_pipeline.py       # Step 6a — Random Forest model
├── xgboost_pipeline.py             # Step 6b — XGBoost model
├── linear_regression_pipeline.py   # Step 6c — Linear Regression model
│
├── waste_dataset.csv               # Raw generated dataset (263,160 rows)
├── cleaned_dataset.csv             # After cleaning (263,160 rows × 23 cols)
├── ml_ready_dataset.csv            # After feature selection (263,160 rows × 15 cols)
│
├── histograms.png                  # EDA — feature distributions
├── heatmap.png                     # EDA — correlation matrix
├── temp_vs_waste.png               # EDA — scatter: temperature vs waste
├── sector_vs_waste.png             # EDA — box plot: sector vs waste
│
├── Random_Forest_Outputs/
│   ├── evaluation_metrics.txt
│   ├── evaluation_metrics.json     # (used by comparison plots)
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── feature_importance.png
│   ├── prediction_distribution.png
│   └── model_metadata.json
│
├── XGBoost_Outputs/
│   ├── evaluation_metrics.txt
│   ├── evaluation_metrics.json
│   ├── predictions.csv
│   ├── feature_importance.csv
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── feature_importance.png
│   └── prediction_distribution.png
│
└── LinearRegression_Outputs/
    ├── evaluation_metrics.txt
    ├── evaluation_metrics.json
    ├── predictions.csv
    ├── feature_importance.csv      # (standardised coefficients)
    ├── actual_vs_predicted.png
    ├── residual_plot.png
    ├── feature_importance.png
    └── prediction_distribution.png
```

---

## 2. Environment Setup

### System Requirements

| Component | Version Used |
|-----------|-------------|
| OS | Windows 10/11 |
| Python | 3.12.3 |
| Java | 21.0.2 (required for PySpark) |

### Install Dependencies

```bash
pip install pyspark
pip install pandas numpy requests
pip install matplotlib seaborn
pip install scikit-learn
pip install xgboost
pip install nbformat
```

### Verify Installation

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("WasteProject").getOrCreate()
print("Spark version:", spark.version)   # Should print 4.1.1
spark.stop()
```

### Windows Note

PySpark on Windows requires `winutils.exe` to write files. The pipelines
handle this automatically by downloading `winutils.exe` for Hadoop 3.3.6
at runtime. No manual setup is needed.

---

## 3. Data Generation

**Script:** `generate_waste_dataset.py`
**Output:** `waste_dataset.csv` (263,160 rows × 10 columns)

### API Used — Open-Meteo Historical Weather API

| Detail | Value |
|--------|-------|
| Endpoint | `https://archive-api.open-meteo.com/v1/archive` |
| Auth | None required (free, no API key) |
| Date range | 2023-01-01 to 2024-12-31 (2 years) |
| Fields fetched | `temperature_2m_max`, `relative_humidity_2m_max`, `weathercode` |
| Cities | 18 major Indian cities |

### Cities Covered

Bangalore, Hyderabad, Chennai, Mumbai, Delhi, Kolkata, Pune, Ahmedabad,
Jaipur, Surat, Lucknow, Nagpur, Indore, Bhopal, Visakhapatnam, Patna,
Coimbatore, Kochi

### Dataset Expansion Strategy

```
18 cities × 730 days × 4 sectors × 5 zones = 263,160 rows
```

Each real API row (city + date + weather) is cross-joined with:
- **4 sectors:** Residential, Commercial, Industrial, Healthcare
- **5 zones:** North, South, East, West, Central

### Waste Amount Formula (Target Column)

```
waste_amount = (population × 0.00005)           ← city-scale base
             + (temperature × random[0.3–0.7])  ← heat drives activity
             + (humidity × random[0.05–0.15])   ← moisture effect
             + (sector_weight × zone_modifier)  ← sector + zone contribution
             + Normal(mean=0, std=20)            ← daily noise
```

**Sector weights** (Industrial > Commercial > Residential > Healthcare):

| Sector | Weight |
|--------|--------|
| Industrial | 350 |
| Commercial | 200 |
| Residential | 120 |
| Healthcare | 80 |

**Zone modifiers:**

| Zone | Modifier |
|------|----------|
| Central | 1.10 |
| East | 1.05 |
| North | 1.00 |
| West | 0.98 |
| South | 0.95 |

### Raw Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `location` | string | City name |
| `date` | string | Date (DD-MM-YYYY) |
| `temperature` | float | Max daily temperature (°C) |
| `humidity` | int | Max daily humidity (%) |
| `weather_condition` | string | Clear / Rain / Clouds etc. |
| `population` | int | City population |
| `zone` | string | City zone |
| `sector` | string | Sector type |
| `pollution_index` | float | Simulated AQI value |
| `waste_amount` | float | **TARGET** — daily waste in tons |

---

## 4. Data Preprocessing & Cleaning

**Script:** `data_cleaning.py`
**Input:** `waste_dataset.csv`
**Output:** `cleaned_dataset.csv` (263,160 rows × 23 columns)

### Cleaning Steps

| Step | Action | Result |
|------|--------|--------|
| 1 | Cast numeric columns to `double` / `int` | Correct types enforced |
| 2 | Parse `date` column → `DateType` (yyyy-MM-dd) | Proper date format |
| 3 | Remove duplicate rows | 0 duplicates found |
| 4 | Drop null rows | 0 nulls found |
| 5 | Validate — assert zero nulls remain | PASSED |

### Feature Engineering (added columns)

| New Column | Description |
|------------|-------------|
| `year` | Extracted from date |
| `month` | Extracted from date (1–12) |
| `day` | Extracted from date (1–31) |
| `day_of_week` | 0=Monday … 6=Sunday |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |
| `season` | Winter / Summer / Rainy / Post-Monsoon |
| `season_enc` | Label encoded (0–3) |
| `sector_enc` | Label encoded (0–3) |
| `zone_enc` | Label encoded (0–4) |
| `location_enc` | Label encoded (0–17) |
| `weather_condition_enc` | Label encoded (0–5) |
| `area_km2` | City area in km² |
| `population_density` | population / area_km2 |

### Waste Amount Fix

The original generation clipped `waste_amount` at 1000 tons. During cleaning,
36,682 capped rows were replaced with a truncated normal distribution
centred at 1050 (range 800–1200) to make the distribution continuous.

### Label Encoding Reference

**sector_enc:** 0=Commercial, 1=Healthcare, 2=Industrial, 3=Residential

**season_enc:** 0=Post-Monsoon, 1=Rainy, 2=Summer, 3=Winter

**zone_enc:** 0=Central, 1=East, 2=North, 3=South, 4=West

**weather_condition_enc:** 0=Clear, 1=Clouds, 2=Drizzle, 3=Heavy Rain,
4=Partly Cloudy, 5=Rain

---

## 5. Feature Selection

**Script:** `feature_selection.py`
**Input:** `cleaned_dataset.csv`
**Output:** `ml_ready_dataset.csv` (263,160 rows × 15 columns)

### Columns Removed

`sector`, `weather_condition`, `location`, `season`, `date`, `area_km2`,
`zone`, `day`

### Columns Kept (ML Features)

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `temperature` | float | Max daily temperature (°C) |
| 2 | `humidity` | int | Max daily humidity (%) |
| 3 | `population` | int | City population |
| 4 | `pollution_index` | float | Simulated AQI |
| 5 | `population_density` | float | People per km² |
| 6 | `year` | int | Year (2023 or 2024) |
| 7 | `month` | int | Month (1–12) |
| 8 | `day_of_week` | int | Day of week (0–6) |
| 9 | `is_weekend` | int | Weekend flag (0/1) |
| 10 | `sector_enc` | int | Sector label encoded |
| 11 | `weather_condition_enc` | int | Weather label encoded |
| 12 | `location_enc` | int | City label encoded |
| 13 | `season_enc` | int | Season label encoded |
| 14 | `zone_enc` | int | Zone label encoded |
| 15 | `waste_amount` | float | **TARGET COLUMN** |

---

## 6. Exploratory Data Analysis (EDA)

**Script:** `eda_analysis.py`
**Input:** `ml_ready_dataset.csv`

### Plots Generated

#### Histograms — `histograms.png`
- 2×3 subplot grid using matplotlib
- 50 bins per histogram
- Columns: temperature, humidity, pollution_index, population_density, waste_amount
- Mean (dashed black) and median (dotted orange) lines overlaid

#### Correlation Heatmap — `heatmap.png`
- Lower-triangle seaborn heatmap
- 14 features including target
- Coolwarm diverging palette, no annotations

#### Scatter Plot — `temp_vs_waste.png`
- X: temperature, Y: waste_amount
- 10,000 sampled points coloured by waste intensity
- Trend line with equation
- Pearson r annotation (r = −0.02 → weak linear relationship)

#### Box Plot — `sector_vs_waste.png`
- X: sector (decoded labels), Y: waste_amount
- Order: Industrial > Commercial > Residential > Healthcare
- Diamond markers show mean per sector

### Key EDA Findings

| Finding | Value |
|---------|-------|
| Population ↔ Waste correlation | +0.8494 (strongest predictor) |
| Population density ↔ Waste | +0.5702 |
| Pollution index ↔ Waste | +0.2883 |
| Temperature ↔ Waste | −0.0201 (weak) |
| Highest waste sector | Industrial (715 tons/day avg) |
| Lowest waste sector | Healthcare (475 tons/day avg) |
| Peak waste month | May (573.60 tons avg) |
| Top waste city | Delhi (1,049 tons/day avg) |

---

## 7. Machine Learning — Model Training

All three models use:
- **Same dataset:** `ml_ready_dataset.csv`
- **Same features:** 14 input columns listed above
- **Same split:** 80% train / 20% test, `random_state=42`
- **Same metrics:** RMSE, MAE, R², MAPE

### Model A — Random Forest Regressor

**Script:** `random_forest_pipeline.py`
**Framework:** PySpark MLlib
**Output folder:** `Random_Forest_Outputs/`

| Parameter | Value |
|-----------|-------|
| `numTrees` | 100 |
| `maxDepth` | 10 |
| `subsamplingRate` | 0.8 |
| `seed` | 42 |
| Training time | ~60 seconds |

Uses `VectorAssembler` to combine features into a single Spark ML vector,
then trains `RandomForestRegressor`. Model metadata saved as JSON.

### Model B — XGBoost Regressor

**Script:** `xgboost_pipeline.py`
**Framework:** scikit-learn API (`xgboost` library)
**Output folder:** `XGBoost_Outputs/`

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 500 |
| `max_depth` | 6 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `reg_alpha` | 0.0 (L1) |
| `reg_lambda` | 1.0 (L2) |
| Training time | ~14 seconds |

### Model C — Linear Regression

**Script:** `linear_regression_pipeline.py`
**Framework:** scikit-learn
**Output folder:** `LinearRegression_Outputs/`

| Parameter | Value |
|-----------|-------|
| `fit_intercept` | True |
| `scaler` | StandardScaler (zero mean, unit variance) |
| Training time | < 1 second |

StandardScaler is applied to prevent large-scale features (e.g. population
in millions) from dominating the coefficients. Scaler is fit on training
data only to prevent data leakage.

---

## 8. Prediction Outputs

Each model saves the following files in its output folder:

| File | Description |
|------|-------------|
| `predictions.csv` | Two columns: `actual`, `predicted` (52,000+ rows) |
| `evaluation_metrics.txt` | Human-readable metrics report |
| `evaluation_metrics.json` | Machine-readable metrics (used by comparison plots) |
| `feature_importance.csv` | Feature name + importance score, sorted descending |

### Sample predictions.csv format

```
actual,predicted
612.39,615.98
666.06,671.48
567.52,579.52
...
```

---

## 9. Visualizations

Each model generates 4 plots saved in its output folder:

| Plot | Filename | Description |
|------|----------|-------------|
| Actual vs Predicted | `actual_vs_predicted.png` | Scatter with perfect-prediction line and R²/RMSE annotation |
| Residual Plot | `residual_plot.png` | Predicted vs (actual−predicted) with ±RMSE bands |
| Feature Importance | `feature_importance.png` | Horizontal bar chart of importance scores |
| Prediction Distribution | `prediction_distribution.png` | Overlapping histograms of actual vs predicted |

Linear Regression's feature importance plot shows **signed standardised
coefficients** (blue = positive effect, red = negative effect) instead of
tree-based importance scores.

When all three model JSON outputs exist, `linear_regression_pipeline.py`
also generates:

| Plot | Filename | Description |
|------|----------|-------------|
| 3-Model Comparison | `3model_comparison.png` | Grouped bar chart: RF vs XGBoost vs LR on all 4 metrics |

---

## 10. Model Comparison Results

| Metric | Random Forest | XGBoost | Linear Regression | Best |
|--------|-------------|---------|-------------------|------|
| RMSE (tons) | 40.51 | **34.88** | 128.33 | XGBoost |
| MAE (tons) | 29.43 | **22.99** | 105.34 | XGBoost |
| R² Score | 0.9781 | **0.9838** | 0.7803 | XGBoost |
| MAPE | 5.70% | **4.34%** | 23.91% | XGBoost |
| Training Time | ~60s | ~14s | **<1s** | Linear Reg |

### Top Feature Importances (all models agree)

| Rank | Feature | RF | XGBoost | LR (|coef|) |
|------|---------|-----|---------|-------------|
| 1 | population | 56.28% | 67.03% | 59.89% |
| 2 | population_density | 21.84% | 13.42% | 19.17% |
| 3 | sector_enc | 10.62% | 9.92% | 0.65% |
| 4 | location_enc | 8.65% | 9.22% | 4.79% |
| 5 | pollution_index | 2.20% | 0.08% | 8.57% |

### Interpretation

- **XGBoost** is the best model overall — highest R² (98.38%) and lowest
  error, while training 4× faster than Random Forest.
- **Random Forest** is a strong second — robust and interpretable, with
  97.81% variance explained.
- **Linear Regression** achieves 78% R² — reasonable for a linear model,
  but the non-linear population-waste relationship limits its accuracy.
  It is useful as a fast baseline and for coefficient interpretability.

---

## 11. How to Run — Step by Step

### Step 1 — Generate Dataset

```bash
python generate_waste_dataset.py
```

Fetches real weather data from Open-Meteo API for 18 cities (2023–2024),
expands to 263,160 rows, saves `waste_dataset.csv`.
Runtime: ~30 seconds (API calls + processing)

---

### Step 2 — Clean Dataset

```bash
python data_cleaning.py
```

Loads `waste_dataset.csv`, fixes types, parses dates, engineers 13 new
features, label-encodes categoricals, saves `cleaned_dataset.csv`.
Runtime: ~20 seconds

---

### Step 3 — Feature Selection

```bash
python feature_selection.py
```

Loads `cleaned_dataset.csv`, drops raw categorical and non-ML columns,
saves `ml_ready_dataset.csv` (15 columns).
Runtime: ~10 seconds

---

### Step 4 — EDA

```bash
python eda_analysis.py
```

Loads `ml_ready_dataset.csv`, generates and saves 4 EDA plots:
`histograms.png`, `heatmap.png`, `temp_vs_waste.png`, `sector_vs_waste.png`.
Runtime: ~15 seconds

---

### Step 5a — Random Forest

```bash
python random_forest_pipeline.py
```

Trains 100-tree Random Forest on 210k rows, evaluates on 52k test rows,
saves all outputs to `Random_Forest_Outputs/`.
Runtime: ~2 minutes

---

### Step 5b — XGBoost

```bash
python xgboost_pipeline.py
```

Trains XGBoost (500 estimators) on same split, saves all outputs to
`XGBoost_Outputs/`.
Runtime: ~15 seconds

---

### Step 5c — Linear Regression

```bash
python linear_regression_pipeline.py
```

Scales features with StandardScaler, trains Linear Regression, saves all
outputs to `LinearRegression_Outputs/`. Also generates the 3-model
comparison chart if RF and XGBoost JSON outputs exist.
Runtime: ~5 seconds

---

### Full Pipeline (run all in order)

```bash
python generate_waste_dataset.py
python data_cleaning.py
python feature_selection.py
python eda_analysis.py
python random_forest_pipeline.py
python xgboost_pipeline.py
python linear_regression_pipeline.py
```

---

## Notes

- All random seeds are fixed at `42` for full reproducibility.
- The dataset is synthetic but grounded in real API weather data — suitable
  for academic ML projects and demonstrations.
- PySpark is used for data loading and Random Forest training; scikit-learn
  and XGBoost handle the other two models.
- All CSV outputs are standard format and can be loaded directly into
  Apache Spark, pandas, or any BI tool.
