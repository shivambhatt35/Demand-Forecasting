# Importing and Installing Libraries

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon
import geopandas as gp
import matplotlib.cm as cm
import itertools
from pathlib import Path
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor = 'dimgray', linewidth=1)
%matplotlib inline

# Declaring Variables 
END_DATE = datetime.today().strftime('%Y-%m-%d')
START_DATE = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')

# Utility Functions

#################################################################################################
# Function Description:
# Computes the average available quantity of each product across snapshots, 
# grouped by platform, warehouse ID, and product ID.

# Input Parameter:
# snapshot_df (pd.DataFrame): A DataFrame containing inventory snapshot data with at least 
# the following columns: 'platform', 'wh_id', 'product_id', and 'available_qty'. 
# Rows may include multiple snapshots per product.

# Output Parameter:
# pd.DataFrame: A DataFrame with one row per unique combination of 
# 'platform', 'wh_id', and 'product_id', including a new column 'avg_availability' 
# that contains the average available quantity for each group.
#################################################################################################

def compute_avg_availability(snapshot_df: pd.DataFrame) -> pd.DataFrame:
    """Computes average availability per product."""
    snapshot_df['available_qty'] = snapshot_df['available_qty'].fillna(0)
    snapshot_df['avg_availability'] = snapshot_df.groupby(
        ['platform', 'wh_id', 'product_id']
    )['available_qty'].transform('mean')
    return snapshot_df.drop_duplicates(subset=['platform', 'wh_id', 'product_id'])

#################################################################################################
# Function Description:
# Filters the polygon mapping data to include only entries where the geographic type is 'Pincode',
# and returns a distinct set of warehouse-to-geo ID mappings.

# Input Parameter:
# polygon_df (pd.DataFrame): A DataFrame containing polygon mapping information with at least
# the columns 'wh_id', 'geo_id', and 'geo_type'.

# Output Parameter:
# pd.DataFrame: A DataFrame containing unique combinations of 'wh_id' and 'geo_id' where 
# 'geo_type' is 'Pincode'.
#################################################################################################

def filter_polygon_mapping(polygon_df: pd.DataFrame) -> pd.DataFrame:
    """Filters polygon mappings by geo_type and ensures required fields are present."""
    polygon_df = polygon_df[polygon_df['geo_type'] == 'Pincode']
    required_cols = ['wh_id', 'geo_id']
    return polygon_df[required_cols].drop_duplicates()


##################################################################################################################################
# Function Description:
# Calculates the proportion of orders placed during each hour of the day, grouped by platform, warehouse, and geographic location.
# This provides hourly weights representing the distribution of orders over 24 hours.

# Input Parameter:
# sales_df (pd.DataFrame): A DataFrame containing order data with at least the following columns:
# - 'order_created_ts' (timestamp of order creation)
# - 'platform' (platform identifier)
# - 'wh_id' (warehouse ID)
# - 'geo_id' (geographic location ID)
# - 'order_id' (unique order identifier)

# Output Parameter:
# pd.DataFrame: A DataFrame with columns ['platform', 'wh_id', 'geo_id', 'order_hour', 'hour_weight'] where
# 'hour_weight' is the fraction of total daily orders occurring in that hour for each platform, warehouse, and geo location.
#################################################################################################################################


def compute_hour_weights(sales_df: pd.DataFrame) -> pd.DataFrame:
    """Computes hour-based weights from order data."""
    sales_df['order_hour'] = pd.to_datetime(sales_df['order_created_ts']).dt.hour

    order_counts = sales_df.groupby(
        ['platform', 'wh_id', 'geo_id', 'order_hour']
    )['order_id'].count().reset_index(name='hourly_orders')

    total_orders = order_counts.groupby(
        ['platform', 'wh_id', 'geo_id']
    )['hourly_orders'].transform('sum')

    order_counts['total_orders'] = total_orders
    order_counts['hour_weight'] = order_counts['hourly_orders'] / total_orders.replace(0, np.nan)

    return order_counts[['platform', 'wh_id', 'geo_id', 'order_hour', 'hour_weight']]


##################################################################################################################################
# Function Description:
# Merges the product assortment availability data with polygon geographic mapping to associate each warehouse and product with a specific geographic area.
# This allows linking inventory data to spatial regions based on warehouse locations.

# Input Parameters:
# - snapshot_df (pd.DataFrame): DataFrame containing product availability snapshots with columns including 'platform', 'wh_id', 'product_id', and 'available_qty'.
# - polygon_df (pd.DataFrame): DataFrame containing polygon mapping data with columns including 'wh_id', 'geo_id', and 'geo_type'.

# Output Parameter:
# pd.DataFrame: A DataFrame resulting from the left join of average availability per product and polygon mapping, filtered to retain only records with valid 'geo_id' values.
##################################################################################################################################

def merge_assortment_with_polygon(snapshot_df: pd.DataFrame, polygon_df: pd.DataFrame) -> pd.DataFrame:
    """Merges assortment master with polygon mapping."""
    filtered_df = compute_avg_availability(snapshot_df)
    polygon_df = filter_polygon_mapping(polygon_df)

    merged_df = pd.merge(filtered_df, polygon_df, how='left', on='wh_id')
    merged_df.dropna(subset=['geo_id'], inplace=True)
    return merged_df

##################################################################################################################################
# Function Description:
# Constructs the foundational dataset for forecasting by merging product assortment availability with polygon geographic mapping
# and enriching it with hourly order weights derived from sales data.

# Input Parameters:
# - assortment_df (pd.DataFrame): Snapshot data containing product availability per warehouse.
# - sales_df (pd.DataFrame): Sales/order data containing timestamps and order information.
# - mapping_df (pd.DataFrame): Polygon mapping data linking warehouses to geographic regions.

# Output Parameter:
# pd.DataFrame: A DataFrame containing platform, warehouse, geo region, product, hourly order weights, and average availability,
# which serves as the base input for forecast modeling.
##################################################################################################################################

def build_forecast_base(assortment_df: pd.DataFrame, sales_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """Builds the base dataframe used for generating forecasts."""
    snapshot_df = merge_assortment_with_polygon(assortment_df, mapping_df)
    hour_weights_df = compute_hour_weights(sales_df)

    base_forecast_df = pd.merge(
        snapshot_df,
        hour_weights_df,
        on=['platform', 'wh_id', 'geo_id'],
        how='left'
    )

    return base_forecast_df[
        ['platform', 'wh_id', 'geo_id', 'product_id', 'order_hour', 'hour_weight', 'avg_availability']
    ]

##################################################################################################################################
# Function Description:
# Prepares the final machine learning dataset by pivoting the hourly order weights to wide format and
# merging average availability for each product and region, ready for training or prediction.

# Input Parameters:
# - snapshot_df (pd.DataFrame): Product availability snapshot data.
# - sales_df (pd.DataFrame): Sales/order data with timestamps.
# - polygon_df (pd.DataFrame): Geographic polygon mapping data.

# Output Parameter:
# pd.DataFrame: A wide-format DataFrame where each row represents a product in a geographic region,
# columns represent hour-based weights for order likelihood, and average availability is included for model targets.
##################################################################################################################################

def prepare_ml_input(snapshot_df: pd.DataFrame, sales_df: pd.DataFrame, polygon_df: pd.DataFrame) -> pd.DataFrame:
    """Prepares the final dataset for ML training or prediction."""
    forecast_df = build_forecast_base(snapshot_df, sales_df, polygon_df)

    # Optional: Pivot or reshape for model input
    ml_df = forecast_df.pivot_table(
        index=['platform', 'wh_id', 'geo_id', 'product_id'],
        columns='order_hour',
        values='hour_weight',
        fill_value=0
    ).reset_index()

    # Merge back average availability for training targets if needed
    avg_availability = forecast_df[
        ['platform', 'wh_id', 'geo_id', 'product_id', 'avg_availability']
    ].drop_duplicates()

    final_ml_input = pd.merge(ml_df, avg_availability, on=['platform', 'wh_id', 'geo_id', 'product_id'])

    return final_ml_input

# Execution Node 1

# ---------------------------- Execution ----------------------------

def load_data_from_db(connection_string: str):
    # Create a SQLAlchemy engine
    engine = create_engine(connection_string)

    # Define your SQL queries to fetch data for each DataFrame
    snapshot_query = "SELECT platform, wh_id, product_id, available_qty FROM snapshot_table"
    sales_query = """
        SELECT platform, wh_id, geo_id, order_created_ts, order_id 
        FROM sales_table
    """
    polygon_query = "SELECT wh_id, geo_id, geo_type FROM polygon_table"

    # Read data into pandas DataFrames
    snapshot_df = pd.read_sql(snapshot_query, engine)
    sales_df = pd.read_sql(sales_query, engine)
    polygon_df = pd.read_sql(polygon_query, engine)

    return snapshot_df, sales_df, polygon_df


if __name__ == "__main__":
    # For 
    connection_string = "postgresql+psycopg2://username:password@host:port/database"

    # Load data from the database
    snapshot_df, sales_df, polygon_df = load_data_from_db(connection_string)

    # Pass the loaded data to your utility function
    final_dataset = prepare_ml_input(snapshot_df, sales_df, polygon_df)

    print(final_dataset.head())

# Utility Functions Part 2

##################################################################################################################################
# Function Description:
#     Generates a DataFrame containing all combinations of dates and hours between the given start and end dates.
#     Each row represents one hour of one day in the date range.

# Input Parameters:
#     start_date (str): The start date in 'YYYY-MM-DD' format.
#     end_date (str): The end date in 'YYYY-MM-DD' format.

# Output:
#     pd.DataFrame: A DataFrame with two columns:
#         - 'local_date': Date part of the datetime (date object).
#         - 'local_hour': Hour of the day (0 to 23).
#     The DataFrame contains rows for every hour in the full date range (inclusive).
##################################################################################################################################

def generate_date_hour_grid(start_date: str, end_date: str) -> pd.DataFrame:
    date_hours = pd.date_range(start=start_date, end=end_date, freq='H')
    df = pd.DataFrame({
        'local_date': date_hours.date,
        'local_hour': date_hours.hour
    })
    return df

##################################################################################################################################
# Function Description:
#     Processes raw orders data to extract local date and hour from the order timestamp,
#     then aggregates the data to count the number of orders for each warehouse, polygon, date, and hour combination.

# Input Parameter:
#     orders_df (pd.DataFrame): DataFrame containing order records with at least the columns:
#         - 'order_ts' (timestamp or string): The timestamp when the order was placed.
#         - 'wh_id' (int or str): Warehouse identifier.
#         - 'polygon_id' (int or str): Polygon/geographic area identifier.

# Output:
#     pd.DataFrame: Aggregated DataFrame grouped by 'wh_id', 'polygon_id', 'local_date', and 'local_hour' with columns:
#         - 'wh_id'
#         - 'polygon_id'
#         - 'local_date' (date object)
#         - 'local_hour' (int, hour of day 0-23)
#         - 'order_count' (int): Number of orders in that grouping.
##################################################################################################################################

def prepare_forecast_base(orders_df: pd.DataFrame) -> pd.DataFrame:
    orders_df['local_date'] = pd.to_datetime(orders_df['order_ts']).dt.date
    orders_df['local_hour'] = pd.to_datetime(orders_df['order_ts']).dt.hour

    grouped = orders_df.groupby(['wh_id', 'polygon_id', 'local_date', 'local_hour']).size().reset_index(name='order_count')
    return grouped

##################################################################################################################################
# Function Description:
#     Creates a complete, dense time grid for all combinations of warehouses and polygons
#     across every hour between start_date and end_date. This ensures that every hour
#     has an entry, even if there were no orders (fills missing hours with zero order count).

# Input Parameter:
#     forecast_base_df (pd.DataFrame): Aggregated order counts grouped by
#         'wh_id', 'polygon_id', 'local_date', and 'local_hour' with an 'order_count' column.
#     start_date (str): The start date of the date-hour grid (format: 'YYYY-MM-DD').
#     end_date (str): The end date of the date-hour grid (format: 'YYYY-MM-DD').

# Output:
#     pd.DataFrame: A DataFrame with all possible combinations of 'wh_id', 'polygon_id',
#         'local_date', and 'local_hour' in the date range, merged with order counts,
#         with missing order counts filled with zero. Columns include:
#         - 'wh_id'
#         - 'polygon_id'
#         - 'local_date'
#         - 'local_hour'
#         - 'order_count' (int)
##################################################################################################################################

def create_dense_time_grid(forecast_base_df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    date_hour_df = generate_date_hour_grid(start_date, end_date)
    unique_combinations = forecast_base_df[['wh_id', 'polygon_id']].drop_duplicates()

    all_combinations = (
        unique_combinations
        .assign(key=1)
        .merge(date_hour_df.assign(key=1), on='key')
        .drop('key', axis=1)
    )

    full_grid_df = all_combinations.merge(
        forecast_base_df,
        on=['wh_id', 'polygon_id', 'local_date', 'local_hour'],
        how='left'
    )

    full_grid_df['order_count'] = full_grid_df['order_count'].fillna(0)
    return full_grid_df

##################################################################################################################################
# Function Description:
#     Adds time-based features to the dataframe to enhance forecasting models.
#     Specifically, it identifies weekends and encodes the hour of the day as cyclical
#     sine and cosine components to capture daily seasonality.

# Input Parameter:
#     df (pd.DataFrame): DataFrame containing at least the columns 'local_date' (date)
#         and 'local_hour' (integer hour of the day, 0-23).

# Output:
#     pd.DataFrame: The input DataFrame enriched with three new columns:
#         - 'is_weekend' (bool): True if the date is Saturday or Sunday, else False.
#         - 'hour_sin' (float): Sine transformation of the hour (for cyclical encoding).
#         - 'hour_cos' (float): Cosine transformation of the hour (for cyclical encoding).
##################################################################################################################################

def enrich_with_features(df: pd.DataFrame) -> pd.DataFrame:
    df['is_weekend'] = pd.to_datetime(df['local_date']).dt.weekday >= 5
    df['hour_sin'] = np.sin(2 * np.pi * df['local_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['local_hour'] / 24)
    return df

##################################################################################################################################
# Function Description:
#     Generates lagged versions of the 'order_count' column for each warehouse and polygon combination.
#     Lag features capture past order counts at specified lag intervals to help in time series forecasting.

# Input Parameter:
#     df (pd.DataFrame): DataFrame containing columns 'wh_id', 'polygon_id', 'local_date', 'local_hour',
#         and 'order_count'.
#     lags (list of int, optional): List of lag periods (in time steps) to generate lag features for.
#         Defaults to [1, 2, 3].

# Output:
#     pd.DataFrame: The input DataFrame augmented with new columns named 'order_count_lag_{lag}',
#         each representing the order count shifted by the corresponding lag period, grouped by 'wh_id' and 'polygon_id'.
 ##################################################################################################################################

def generate_lag_features(df: pd.DataFrame, lags=[1, 2, 3]) -> pd.DataFrame:
    df = df.sort_values(['wh_id', 'polygon_id', 'local_date', 'local_hour'])
    for lag in lags:
        df[f'order_count_lag_{lag}'] = df.groupby(['wh_id', 'polygon_id'])['order_count'].shift(lag)
    return df

##################################################################################################################################
# Function Description:
#     Fills missing (NaN) values in all lag feature columns with zeros.
#     This ensures that lag features have no missing values, which is important for model training or further processing.

# Input Parameter:
#     df (pd.DataFrame): DataFrame containing one or more columns with 'lag' in their names, representing lag features.

# Output:
#     pd.DataFrame: The input DataFrame with missing values in all lag feature columns replaced by 0.
##################################################################################################################################

def fill_missing_lags(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [col for col in df.columns if 'lag' in col]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df

# Execution Part 2

# ------------------------
# Main Pipeline
# ------------------------
def main_pipeline(orders_df: pd.DataFrame) -> pd.DataFrame:
    forecast_base = prepare_forecast_base(orders_df)
    dense_grid = create_dense_time_grid(forecast_base, START_DATE, END_DATE)
    enriched = enrich_with_features(dense_grid)
    lagged = generate_lag_features(enriched)
    final_df = fill_missing_lags(lagged)
    return final_df

# Remove for security reason
DB_TYPE = 'xxxxxx' 
DB_DRIVER = 'xxxxx'
DB_USER = 'xxxxx'
DB_PASS = 'xxxxx'
DB_HOST = 'xxxxx'
DB_PORT = 'xxxxx'
DB_NAME = 'xxxxx'

if __name__ == '__main__':
    # Construct connection string
    connection_string = f"{DB_TYPE}+{DB_DRIVER}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def read_orders_from_db(start_date: str, end_date: str) -> pd.DataFrame:
    # Create SQLAlchemy engine
    engine = create_engine(connection_string)

    query = f"""
    SELECT order_ts, wh_id, polygon_id
    FROM orders_table
    WHERE order_ts >= '{start_date}' AND order_ts < '{end_date}'
    """

    df = pd.read_sql(query, con=engine)
    return df

if __name__ == '__main__':

    today = datetime.today().date()
    start_date = today - timedelta(days=30)
    end_date = today

    # Read data from DB
    orders_df = read_orders_from_db(str(start_date), str(end_date))

    processed_df = main_pipeline(orders_df)
    print(processed_df.head())
