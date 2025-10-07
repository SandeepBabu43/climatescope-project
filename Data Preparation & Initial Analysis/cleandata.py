import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- 1. Load and Initial Inspection ---
file_path = 'GlobalWeatherRepository_cleaned.csv'
df = pd.read_csv(file_path)

print("--- 1. Initial Dataset Structure and Data Types (df.info()) ---")
df.info()

print("\n--- 1. First 5 rows of the dataset (df.head()) ---")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))


# --- 2. Identify Missing Values and Data Coverage ---
print("\n--- 2. Missing Values Count per Column ---")
missing_values = df.isnull().sum()
print(missing_values[missing_values > 0].sort_values(ascending=False).to_markdown())

# For simplicity in this script, we'll assume the coverage is sufficient based on the counts.
# Anomalies would typically be checked using descriptive statistics (df.describe())
# and visualization (box plots), but for a text-based code deliverable, we focus on
# identifying data type inconsistencies and obvious missingness.


# --- 3. Handle Missing or Inconsistent Entries ---

# A. Convert date/time columns to datetime objects
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['sunrise'] = pd.to_datetime(df['sunrise'], format='%I:%M %p', errors='coerce')
df['sunset'] = pd.to_datetime(df['sunset'], format='%I:%M %p', errors='coerce')
df['moonrise'] = pd.to_datetime(df['moonrise'], format='%I:%M %p', errors='coerce')
df['moonset'] = pd.to_datetime(df['moonset'], format='%I:%M %p', errors='coerce')

# B. Handle Missing Numerical Data (Imputation)
# Identify numerical columns (excluding epoch, latitude/longitude, and a few others that should not be mean-imputed)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

# Columns to exclude from mean imputation (as they are identifiers or unique time stamps)
cols_to_exclude = ['last_updated_epoch', 'latitude', 'longitude']
numerical_impute_cols = [col for col in numerical_cols if col not in cols_to_exclude]

print("\n--- 3. Handling Missing Numerical Data (Mean Imputation) ---")
# Impute missing values in the selected numerical columns with the mean
for col in numerical_impute_cols:
    df[col].fillna(df[col].mean(), inplace=True)
    if missing_values[col] > 0:
        print(f"Filled {missing_values[col]} missing values in '{col}' with the mean: {df[col].mean():.2f}")


# --- 4. Convert Units and Normalize Values ---

# A. Unit Conversion (Removing redundant columns)
# The dataset has both C/F and km/miles. We will keep Celsius and Kilometers for a consistent metric system.
df.drop(columns=['temperature_fahrenheit', 'feels_like_fahrenheit', 'visibility_miles', 'pressure_in', 'precip_in', 'wind_mph', 'gust_mph'], inplace=True)
print("\n--- 4. Unit Conversion: Removed redundant unit columns (Fahrenheit, Miles, Inches) ---")

# B. Normalization (Min-Max Scaling)
# Normalize 'temperature_celsius' to a range of 0 to 1
scaler = MinMaxScaler()
df['temperature_celsius_normalized'] = scaler.fit_transform(df[['temperature_celsius']])
print("\n--- 4. Normalization: 'temperature_celsius' normalized and saved as 'temperature_celsius_normalized'. ---")


# --- 5. Aggregate or Filter Data ---

# A. Filtering: Example - Filter for a single country
country_filter = 'India'
df_filtered = df[df['country'] == country_filter].copy()
print(f"\n--- 5. Filtering: Dataset filtered for '{country_filter}'. New size: {len(df_filtered)} rows. ---")

# B. Aggregation: Example - Daily to Monthly Average Temperature for the filtered data
# Extract the month from the 'last_updated' datetime object
df_filtered['month'] = df_filtered['last_updated'].dt.to_period('M')

# Group by month and calculate the mean of key numerical variables
monthly_average = df_filtered.groupby('month').agg(
    avg_temp_c=('temperature_celsius', 'mean'),
    avg_humidity=('humidity', 'mean'),
    max_wind_kph=('wind_kph', 'max'),
    count=('country', 'size')
).reset_index()

print(f"\n--- 5. Aggregation: Monthly average temperature and other metrics for '{country_filter}'. (First 5 rows) ---")
print(monthly_average.head().to_markdown(index=False, numalign="left", stralign="left"))

# --- 6. Evaluation and Deliverable ---

# Save the final cleaned and preprocessed dataset
output_file_name = 'GlobalWeatherRepository_cleaned_processed.csv'
df.to_csv(output_file_name, index=False)
print(f"\n--- 6. Deliverable: Cleaned dataset saved as '{output_file_name}' ---")


# Generate Summary Document Outline (Programmatic Output)
print("\n\n--- SUMMARY DOCUMENT OUTLINE: Data Schema and Quality Issues ---")
print("1. Data Schema (Final Cleaned Dataset):")
df.info()
print("\n2. Key Variables (Post-Processing):")
print("- 'temperature_celsius' (Original): Mean, Min, Max:")
print(df['temperature_celsius'].agg(['mean', 'min', 'max']).to_markdown())
print("\n- 'temperature_celsius_normalized' (New Normalized Column): Mean, Min, Max:")
print(df['temperature_celsius_normalized'].agg(['mean', 'min', 'max']).to_markdown())
print("\n- Categorical Variable Counts (Top 5 Conditions):")
print(df['condition_text'].value_counts().head().to_markdown())
print("\n3. Data Quality Issues Handled:")
print("- Missing Values: Filled all missing numerical entries with the column mean.")
print("- Unit Inconsistency: Removed redundant columns (Fahrenheit, Miles, Inches) to standardize on Metric (Celsius, km, mb).")
print("- Data Types: Converted 'last_updated' and all time columns ('sunrise', 'sunset', etc.) to proper datetime objects.")
