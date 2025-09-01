# Demand Forecasting with Time-Series Analysis

A robust, modular pipeline for **hourly demand prediction** using time-series analysis and machine learning.  
Built for accurate **inventory management**, **supply chain optimization**, and **logistics forecasting**.

## Project Summary

This project focuses on building a robust time-series forecasting pipeline for **demand prediction**, incorporating both statistical analysis and predictive analytics. Leveraging historical order and product availability data, the pipeline prepares granular hourly-level forecasts by warehouse and delivery polygons (geographical areas).

The core of the project lies in transforming raw transaction and inventory snapshots into enriched datasets that include temporal features (like hour-of-day seasonality, weekend indicators) and lag-based historical demand patterns. These features enable downstream machine learning models or statistical forecasting methods to **capture patterns and trends effectively**, **improving inventory managemen**t, **supply chain optimization**, and **demand planning accuracy**.

The solution integrates **data preparation**, **feature engineering**, and **time-based interpolation** to create a dense grid of order counts by hour and geography — filling missing time slots with zeros and generating lagged variables to capture autocorrelation in demand.

---

## Key Features

| Feature   | Description |
|-----------|-------------|
| **Hourly Forecasting** | Aggregates order counts at the hourly level for each warehouse and polygon. |
| **Dense Time Grid** | Builds a continuous hourly grid ensuring no missing time slots. |
| **Feature Enrichment** | Adds temporal features like `weekend flags` and `hour-of-day encodings`. |
| **Lag Generation** | Computes lag features to capture demand autocorrelation (e.g., t-1, t-2, t-3). |
| **Availability & Geo Mapping** | Merges product availability with delivery zone data. |
| **Flexible Integration** | Reads from SQL databases (PostgreSQL / MySQL) and integrates with pipelines. |
| **Missing Data Handling** | Fills missing orders and lags for consistent datasets. |

---

## How It Works

1.  **Data Extraction**  
   Pulls order and product availability data from SQL sources based on selected date ranges.

2.  **Data Aggregation**  
   Aggregates orders by `warehouse`, `polygon`, `date`, and `hour`.

3.  **Dense Grid Creation**  
   Generates a complete grid of all hour-warehouse-polygon combinations; fills missing time slots with `0`.

4.  **Feature Engineering**  
   - Adds weekend flags (`is_weekend`)  
   - Adds cyclical features: `sin(hour)`, `cos(hour)`

5.  **Lag Feature Computation**  
   - Lag-1, Lag-2, Lag-3 order count features.
   - Enables short-term trend modeling.

6.  **Final Dataset**  
   Output dataset is fully enriched, cleaned, and structured for **ML training or forecasting**.

---

## Tech Stack

| Tool / Library | Purpose |
|----------------|---------|
| **Python 3.x** | Core language for the entire ETL and modeling pipeline |
| **Pandas / NumPy** | Data manipulation and numerical computation |
| **SQLAlchemy / pandas.read_sql** | Extracting data from relational databases |
| **datetime module** | Handling and engineering time-based features |
| **PostgreSQL / MySQL / RDBMS** | Source systems for historical data |
| **scikit-learn / statsmodels / Prophet** | Time-series and ML model training |

---

## Use Cases

- **Inventory Demand Planning**
- **Retail & E-Commerce Forecasting**
- **Multi-Geo Delivery Optimization**
- **Supply Chain Load Balancing**

---
