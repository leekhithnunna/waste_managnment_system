# ♻️ Smart City Waste Prediction System

> End-to-end Machine Learning pipeline for predicting daily waste generation across 18 major Indian cities using real weather API data, Apache Spark, and XGBoost.

---

## 🚀 Live Dashboard

```bash
cd Framework
python -m streamlit run dashboard/app.py
```

---

## 📁 Project Structure

```
├── generate_waste_dataset.py       # Fetch Open-Meteo API + generate 263k row dataset
├── data_cleaning.py                # Spark cleaning + feature engineering
├── feature_selection.py            # Select ML-ready features
├── eda_analysis.py                 # EDA plots (histograms, heatmap, scatter, box)
├── visualization.py                # Spark-based visualizations
│
├── random_forest_pipeline.py       # Random Forest  → R² = 0.9781
├── xgboost_pipeline.py             # XGBoost        → R² = 0.9838 ✅ Best
├── linear_regression_pipeline.py   # Linear Reg     → R² = 0.7803
│
├── ml_ready_dataset.csv            # 263,160 rows × 15 features
│
├── Framework/                      # Prediction system + Streamlit dashboard
│   ├── dashboard/app.py            # Interactive dashboard
│   ├── src/                        # predict, simulate, visualize, utils
│   ├── models/waste_model.pkl      # Trained XGBoost model
│   └── outputs/                    # Model evaluation plots
│
├── Random_Forest_Outputs/          # RF metrics, plots, predictions
├── XGBoost_Outputs/                # XGBoost metrics, plots, predictions
├── LinearRegression_Outputs/       # LR metrics, plots, predictions
│
└── PROJECT_README.md               # Full project documentation
```

---

## 📊 Model Results

| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|----|------|
| **XGBoost** | **34.88** | **22.99** | **0.9838** | **4.34%** |
| Random Forest | 40.51 | 29.43 | 0.9781 | 5.70% |
| Linear Regression | 128.33 | 105.34 | 0.7803 | 23.91% |

---

## ⚙️ Setup

```bash
pip install pyspark pandas numpy matplotlib seaborn scikit-learn xgboost streamlit joblib requests
```

### Run Pipeline (in order)

```bash
python generate_waste_dataset.py      # Step 1 — Generate dataset
python data_cleaning.py               # Step 2 — Clean data
python feature_selection.py           # Step 3 — Select features
python eda_analysis.py                # Step 4 — EDA
python random_forest_pipeline.py      # Step 5a — Train RF
python xgboost_pipeline.py            # Step 5b — Train XGBoost
python linear_regression_pipeline.py  # Step 5c — Train LR
```

---

## 🌐 API Used

**Open-Meteo Historical Weather API** — free, no key required  
`https://archive-api.open-meteo.com/v1/archive`  
Fetches: temperature, humidity, weather codes for 18 Indian cities (2023–2024)

---

## 🏙️ Cities Covered

Bangalore · Hyderabad · Chennai · Mumbai · Delhi · Kolkata · Pune · Ahmedabad  
Jaipur · Surat · Lucknow · Nagpur · Indore · Bhopal · Visakhapatnam · Patna · Coimbatore · Kochi

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
