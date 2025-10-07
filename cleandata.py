import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('GlobalWeatherRepository.csv')

# --- Initial Analysis ---
# Inspecting for missing values (Initial check showed no missing values)
# print(df.isnull().sum().any())

# --- Data Cleaning and Preprocessing (Tasks 3 & 4) ---

# Convert 'last_updated' to datetime and create a monthly period column for aggregation
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['observation_month'] = df['last_updated'].dt.to_period('M')

# Identify and select columns to keep (metric units and identifiers)
cols_to_keep = [
    'country', 'location_name', 'latitude', 'longitude', 'timezone',
    'observation_month', 'temperature_celsius', 'condition_text',
    'wind_kph', 'wind_degree', 'wind_direction', 'pressure_mb',
    'precip_mm', 'humidity', 'cloud', 'feels_like_celsius',
    'visibility_km', 'uv_index', 'gust_kph',
    'air_quality_Carbon_Monoxide', 'air_quality_Ozone', 'air_quality_Nitrogen_dioxide',
    'air_quality_Sulphur_dioxide', 'air_quality_PM2.5', 'air_quality_PM10',
    'air_quality_us-epa-index', 'air_quality_gb-defra-index',
    'sunrise', 'sunset', 'moonrise', 'moonset', 'moon_phase', 'moon_illumination'
]
df_cleaned = df[cols_to_keep]

# --- Aggregation (Task 5) ---

# Define aggregation functions
agg_funcs = {
    # Identifier/Categorical columns (take the first or most frequent value)
    'latitude': 'first', 'longitude': 'first', 'timezone': 'first',
    'condition_text': lambda x: x.mode()[0] if not x.mode().empty else np.nan, # Most frequent condition
    'wind_direction': lambda x: x.mode()[0] if not x.mode().empty else np.nan, # Most frequent wind direction
    'sunrise': 'first', 'sunset': 'first', 'moonrise': 'first',
    'moonset': 'first', 'moon_phase': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
    # Numerical columns (take the mean, except precipitation which is summed)
    'temperature_celsius': 'mean', 'wind_kph': 'mean', 'wind_degree': 'mean',
    'pressure_mb': 'mean', 'precip_mm': 'sum', 'humidity': 'mean',
    'cloud': 'mean', 'feels_like_celsius': 'mean', 'visibility_km': 'mean',
    'uv_index': 'mean', 'gust_kph': 'mean', 'moon_illumination': 'mean',
    'air_quality_Carbon_Monoxide': 'mean', 'air_quality_Ozone': 'mean',
    'air_quality_Nitrogen_dioxide': 'mean', 'air_quality_Sulphur_dioxide': 'mean',
    'air_quality_PM2.5': 'mean', 'air_quality_PM10': 'mean',
    'air_quality_us-epa-index': lambda x: x.mode()[0] if not x.mode().empty else np.nan, # Most frequent index
    'air_quality_gb-defra-index': lambda x: x.mode()[0] if not x.mode().empty else np.nan  # Most frequent index
}

# Group by country, location, and month to calculate monthly averages
df_monthly_avg = df_cleaned.groupby(['country', 'location_name', 'observation_month'], as_index=False).agg(agg_funcs)

# Rename columns to reflect the aggregation
new_columns = {col: f'avg_{col}' for col in df_monthly_avg.columns if col not in ['country', 'location_name', 'observation_month', 'timezone', 'condition_text', 'wind_direction', 'sunrise', 'sunset', 'moonrise', 'moonset', 'moon_phase', 'air_quality_us-epa-index', 'air_quality_gb-defra-index']}
df_monthly_avg = df_monthly_avg.rename(columns=new_columns)
df_monthly_avg = df_monthly_avg.rename(columns={'precip_mm': 'total_precip_mm'})

# Save the cleaned and preprocessed dataset
output_filename = 'GlobalWeatherRepository_Monthly_Avg.csv'
df_monthly_avg.to_csv(output_filename, index=False)