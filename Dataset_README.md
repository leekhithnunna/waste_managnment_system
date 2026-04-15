# Smart City Waste Prediction — Dataset README

## Overview

This README documents how the `waste_dataset.csv` was generated for the
**Smart City Waste Prediction** project. The dataset is designed to be
ML-ready and compatible with Apache Spark MLlib for training waste
prediction models across major Indian cities.

---

## Files

| File | Description |
|------|-------------|
| `generate_waste_dataset.py` | Script that fetches real API data and generates the raw dataset |
| `clean_waste_dataset.py` | Script that cleans, transforms, and feature-engineers the raw dataset |
| `waste_dataset.csv` | Raw generated dataset (263,160 rows × 10 columns) |
| `waste_dataset_cleaned.csv` | Final ML-ready cleaned dataset (263,160 rows × 23 columns) |

---

## API Used — Open-Meteo Historical Weather API

### What is Open-Meteo?

[Open-Meteo](https://open-meteo.com) is a free, open-source weather API
that provides historical and forecast weather data globally. It requires
**no API key**, no registration, and has no rate-limit for reasonable usage.

### Endpoint Used

```
GET https://archive-api.open-meteo.com/v1/archive
```

### Parameters Sent Per City

| Parameter | Value | Description |
|-----------|-------|-------------|
| `latitude` | e.g. `12.97` | City latitude |
| `longitude` | e.g. `77.59` | City longitude |
| `start_date` | `2023-01-01` | Start of date range |
| `end_date` | `2024-12-31` | End of date range (~2 years) |
| `daily` | `temperature_2m_max, relative_humidity_2m_max, weathercode` | Daily aggregates requested |
| `timezone` | `Asia/Kolkata` | IST timezone for Indian cities |

### Fields Returned by API

| API Field | Mapped Column | Description |
|-----------|---------------|-------------|
| `time` | `date` | Date in YYYY-MM-DD |
| `temperature_2m_max` | `temperature` | Max daily temperature at 2m height (°C) |
| `relative_humidity_2m_max` | `humidity` | Max daily relative humidity (%) |
| `weathercode` | `weather_condition` | WMO weather interpretation code (mapped to text) |

### WMO Weather Code Mapping

Open-Meteo returns numeric WMO codes. These were mapped to human-readable
labels as follows:

| WMO Code(s) | Mapped Label |
|-------------|--------------|
| 0, 1 | Clear |
| 2 | Partly Cloudy |
| 3 | Clouds |
| 45, 48 | Fog |
| 51, 53, 55 | Drizzle |
| 61, 63 | Rain |
| 65, 82 | Heavy Rain |
| 80, 81 | Rain |
| 95, 96, 99 | Thunderstorm |

### Why Open-Meteo?

- Completely free — no API key or account needed
- Returns clean daily aggregates (no parsing overhead)
- Covers all Indian cities by lat/lon coordinates
- Reliable uptime and consistent JSON response format
- Supports historical data going back decades

---

## Cities Covered

18 major Indian cities were selected to ensure geographic and demographic diversity:

| City | Latitude | Longitude | Population | Area (km²) |
|------|----------|-----------|------------|------------|
| Bangalore | 12.97 | 77.59 | 13,190,000 | 741 |
| Hyderabad | 17.38 | 78.47 | 10,350,000 | 650 |
| Chennai | 13.08 | 80.27 | 9,110,000 | 426 |
| Mumbai | 19.07 | 72.87 | 20,670,000 | 603 |
| Delhi | 28.61 | 77.20 | 32,940,000 | 1484 |
| Kolkata | 22.57 | 88.36 | 14,850,000 | 185 |
| Pune | 18.52 | 73.85 | 7,280,000 | 331 |
| Ahmedabad | 23.02 | 72.57 | 8,450,000 | 505 |
| Jaipur | 26.91 | 75.79 | 3,950,000 | 467 |
| Surat | 21.17 | 72.83 | 7,180,000 | 395 |
| Lucknow | 26.85 | 80.95 | 3,680,000 | 349 |
| Nagpur | 21.14 | 79.08 | 2,900,000 | 227 |
| Indore | 22.72 | 75.86 | 3,500,000 | 530 |
| Bhopal | 23.26 | 77.41 | 2,400,000 | 463 |
| Visakhapatnam | 17.69 | 83.22 | 2,100,000 | 682 |
| Patna | 25.59 | 85.13 | 2,350,000 | 136 |
| Coimbatore | 11.00 | 76.96 | 2,150,000 | 246 |
| Kochi | 9.93 | 76.26 | 2,120,000 | 94 |

---

## Dataset Generation — Step by Step

### Step 1 — Fetch Real Weather Data

For each of the 18 cities, the script calls the Open-Meteo API and
retrieves daily weather records for the full 2-year period
(2023-01-01 to 2024-12-31), giving approximately 730 rows per city
and **13,158 real API rows** in total.

A 0.5-second delay is added between each city request to be polite
to the API server.

### Step 2 — Handle Missing API Values

Any missing temperature or humidity values are filled using the
city-level median (grouped by location). Missing weather codes
default to `0` (Clear).

### Step 3 — Expand Rows via Cross Join

Each (city × date) row is cross-joined with:
- 4 sectors: `Residential`, `Commercial`, `Industrial`, `Healthcare`
- 5 zones: `North`, `South`, `East`, `West`, `Central`

This multiplies the base 13,158 rows by 20, producing **263,160 rows**.

```
18 cities × 730 days × 4 sectors × 5 zones = 263,160 rows
```

### Step 4 — Simulate pollution_index (AQI Proxy)

Each city has a realistic base AQI value (e.g., Delhi = 180, Kochi = 65).
Daily variation is added using:

```
pollution_index = city_base_AQI
               + Normal(mean=0, std=15)        # daily random noise
               + (humidity - 60) × 0.3         # humidity correlation
```

Values are clipped to the range [20, 500].

### Step 5 — Generate waste_amount (Target Column)

The target column is generated using a realistic multi-factor formula:

```
waste_amount = (population × 0.00005)           # city-scale base load
             + (temperature × random[0.3–0.7])  # heat drives activity
             + (humidity × random[0.05–0.15])   # moisture effect
             + (sector_weight × zone_modifier)  # sector + zone contribution
             + Normal(mean=0, std=20)            # daily noise
```

**Sector weights** enforce the ordering Industrial > Commercial > Residential > Healthcare:

| Sector | Weight |
|--------|--------|
| Industrial | 350 |
| Commercial | 200 |
| Residential | 120 |
| Healthcare | 80 |

**Zone modifiers** reflect density differences within a city:

| Zone | Modifier |
|------|----------|
| Central | 1.10 |
| East | 1.05 |
| North | 1.00 |
| West | 0.98 |
| South | 0.95 |

Final values are clipped to [50, 1000] tons/day and rounded to 2 decimal places.

### Step 6 — Shuffle and Save

Rows are randomly shuffled (seed=42) before saving to prevent any
ordering bias during ML training.

---

## Dataset Schema — waste_dataset.csv (Raw)

| Column | Type | Description |
|--------|------|-------------|
| `location` | string | City name |
| `date` | string | Date (DD-MM-YYYY in raw, YYYY-MM-DD after cleaning) |
| `temperature` | float | Max daily temperature in °C |
| `humidity` | int | Max daily relative humidity (%) |
| `weather_condition` | string | Human-readable weather label |
| `population` | int | City population (static per city) |
| `zone` | string | City zone (North/South/East/West/Central) |
| `sector` | string | Sector type |
| `pollution_index` | float | Simulated AQI value |
| `waste_amount` | float | Daily waste in tons (TARGET) |

---

## Dataset Schema — waste_dataset_cleaned.csv (ML-Ready)

23 columns total after cleaning and feature engineering:

| Column | Type | Description |
|--------|------|-------------|
| `location` | string | City name (raw) |
| `date` | string | Date in YYYY-MM-DD |
| `temperature` | float32 | Max daily temperature (°C) |
| `humidity` | int16 | Max daily humidity (%) |
| `weather_condition` | string | Weather label (raw) |
| `population` | int64 | City population |
| `area_km2` | float32 | City area in km² |
| `population_density` | float32 | population / area_km2 |
| `zone` | string | City zone (raw) |
| `zone_enc` | int16 | Zone label encoded (0–4) |
| `sector` | string | Sector (raw) |
| `sector_enc` | int16 | Sector label encoded (0–3) |
| `pollution_index` | float32 | Simulated AQI |
| `year` | int16 | Extracted year |
| `month` | int8 | Extracted month (1–12) |
| `day` | int8 | Extracted day (1–31) |
| `day_of_week` | int8 | 0=Monday … 6=Sunday |
| `is_weekend` | int8 | 1 if Saturday or Sunday, else 0 |
| `season` | string | Season label (raw) |
| `season_enc` | int16 | Season label encoded (0–3) |
| `weather_condition_enc` | int16 | Weather label encoded (0–5) |
| `location_enc` | int16 | City label encoded (0–17) |
| `waste_amount` | float32 | Daily waste in tons — TARGET COLUMN |

---

## Label Encoding Reference

### sector_enc
| Code | Label |
|------|-------|
| 0 | Commercial |
| 1 | Healthcare |
| 2 | Industrial |
| 3 | Residential |

### weather_condition_enc
| Code | Label |
|------|-------|
| 0 | Clear |
| 1 | Clouds |
| 2 | Drizzle |
| 3 | Heavy Rain |
| 4 | Partly Cloudy |
| 5 | Rain |

### season_enc
| Code | Label |
|------|-------|
| 0 | Post-Monsoon |
| 1 | Rainy |
| 2 | Summer |
| 3 | Winter |

### zone_enc
| Code | Label |
|------|-------|
| 0 | Central |
| 1 | East |
| 2 | North |
| 3 | South |
| 4 | West |

### location_enc
| Code | City |
|------|------|
| 0 | Ahmedabad |
| 1 | Bangalore |
| 2 | Bhopal |
| 3 | Chennai |
| 4 | Coimbatore |
| 5 | Delhi |
| 6 | Hyderabad |
| 7 | Indore |
| 8 | Jaipur |
| 9 | Kochi |
| 10 | Kolkata |
| 11 | Lucknow |
| 12 | Mumbai |
| 13 | Nagpur |
| 14 | Patna |
| 15 | Pune |
| 16 | Surat |
| 17 | Visakhapatnam |

---

## Data Quality Summary

| Check | Result |
|-------|--------|
| Total rows | 263,160 |
| Missing values | 0 |
| Duplicate rows | 0 |
| Date range | 2023-01-01 to 2024-12-31 |
| Cities covered | 18 |
| Sectors | 4 |
| Zones per city | 5 |
| waste_amount range | 131.58 – 1200.00 tons/day |

---

## How to Reproduce

### Requirements

```bash
pip install requests pandas numpy
```

### Run Generation

```bash
python generate_waste_dataset.py
```

This fetches live data from the Open-Meteo API and saves `waste_dataset.csv`.

### Run Cleaning

```bash
python clean_waste_dataset.py
```

This reads `waste_dataset.csv` and saves `waste_dataset_cleaned.csv`.

---

## Usage in Apache Spark

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("WastePrediction").getOrCreate()

df = spark.read.csv("waste_dataset_cleaned.csv", header=True, inferSchema=True)
df.printSchema()
df.show(5)
```

Feature columns for MLlib:

```python
feature_cols = [
    "temperature", "humidity", "pollution_index",
    "population", "population_density",
    "sector_enc", "zone_enc", "location_enc",
    "weather_condition_enc", "season_enc",
    "year", "month", "day", "day_of_week", "is_weekend"
]
label_col = "waste_amount"
```

---

## Notes

- Population values are static per city (census-approximate figures).
- `waste_amount` values above 1000 tons (originally capped) were replaced
  during cleaning with a truncated normal distribution centred at 1050
  (range 800–1200) to ensure a continuous, realistic distribution.
- The dataset is intended for supervised regression tasks.
- Random seeds (42 for generation, 99 for cleaning) are fixed for reproducibility.
