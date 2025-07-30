# Demand Forecasting
This project focuses on building a robust time-series forecasting pipeline for demand prediction, incorporating both statistical analysis and predictive analytics. Leveraging historical order and product availability data, the pipeline prepares granular hourly-level forecasts by warehouse and delivery polygons (geographical areas).

The core of the project lies in transforming raw transaction and inventory snapshots into enriched datasets that include temporal features (like hour-of-day seasonality, weekend indicators) and lag-based historical demand patterns. These features enable downstream machine learning models or statistical forecasting methods to capture patterns and trends effectively, improving inventory management, supply chain optimization, and demand planning accuracy.

The solution integrates data preparation, feature engineering, and time-based interpolation to create a dense grid of order counts by hour and geography — filling missing time slots with zeros and generating lagged variables to capture autocorrelation in demand.

**Key Features:**<br>
1. Hourly Granular Forecasting Base: Aggregates order counts at the hourly level for each warehouse and polygon.<br>
2. Dense Time Grid Creation: Builds a continuous time series grid covering all hours within the target date range, ensuring no time gaps.<br>
3. Feature Enrichment: Adds temporal features such as weekend flags and cyclical encoding of hours using sine and cosine transformations.<br>
4. Lag Feature Generation: Computes lagged order counts to capture temporal dependencies and trends in demand.<br>
5. Availability and Polygon Mapping: Merges product availability data with geographic mappings for comprehensive forecasting inputs.<br>
6. Flexible Data Source Integration: Reads data directly from databases, allowing seamless integration into existing data pipelines.<br>
7. Clean Handling of Missing Data: Fills missing order counts and lag features to maintain data consistency.<br>

**How It Works:**<br>
1. Data Extraction: The system pulls order and availability data from source databases, filtering by date ranges as needed.<br>
2. Data Aggregation: Orders are aggregated by warehouse, polygon, date, and hour to get order counts per time slot.<br>
3. Dense Grid Construction: A complete hourly time grid is generated for all warehouse-polygon combinations, filling in any missing hour data with zeros.<br>
4. Feature Engineering: Additional features like weekend indicators and cyclical hour encodings are added to capture time-based patterns.<br>
5. Lag Computation: Lag features (e.g., previous 1, 2, and 3 hour order counts) are created to capture short-term temporal autocorrelations.<br>
6. Final Dataset Preparation: The enriched dataset is prepared for downstream predictive modeling or statistical analysis.<br>

**Technology Used:**<br>
1. Python 3.x — Core programming language for ETL and feature engineering.<br>
2. Pandas & NumPy — Data manipulation, aggregation, and numerical computations.<br>
3. SQLAlchemy & Pandas.read_sql — Database connection and data extraction.<br>
4. Datetime Module — Handling and generation of date/time features.<br>
5. PostgreSQL / MySQL / Other RDBMS — Source database for order and inventory data.<br>
6. Scikit-learn / Statsmodels / Prophet / Other ML Frameworks — For building forecasting models using the prepared dataset.
