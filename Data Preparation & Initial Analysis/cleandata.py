import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from google.colab import files
import io

# ====================================================================
# STEP 0: FILE UPLOAD
# ====================================================================
print("Please upload the 'GlobalWeatherRepository.csv' file when prompted.")
uploaded = files.upload()

# Determine the file name and load the data
file_path = list(uploaded.keys())[0] # Gets the name of the uploaded file
df = pd.read_csv(io.BytesIO(uploaded[file_path]))

print(f"\nSuccessfully loaded dataset: {file_path}")
print("====================================================================")


# ====================================================================
# STEP 1: INSPECTION (Structure, Data Types, Missing Values)
# ====================================================================
print("--- 1. Initial Dataset Structure and Data Types (df.info()) ---")
df.info()

print("\n--- 1. First 5 rows of the dataset (df.head()) ---")
print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

print("\n--- 1. Missing Values Count per Column ---")
missing_values = df.isnull().sum()
# Print only columns with missing data
print(missing_values[missing_values > 0].sort_values(ascending=False).to_markdown())
# Store the initial missing counts before imputation
initial_missing_counts = missing_values[missing_values > 0]


# ====================================================================
# STEP 2: CLEANING & HANDLING (Missing Data, Inconsistent Entries, Units)
# ====================================================================

# A. Handle Data Types (Convert date/time columns)
df['last_updated'] = pd.to_datetime(df['last_updated'])
# Use errors='coerce' to turn invalid time strings into NaT (Not a Time)
time_cols = ['sunrise', 'sunset', 'moonrise', 'moonset']
for col in time_cols:
    df[col] = pd.to_datetime(df[col], format='%I:%M %p', errors='coerce')
print("\n--- 2A. Data Types: Date/Time columns converted to datetime objects. ---")

# B. Handle Missing Numerical Data (Mean Imputation)
numerical_impute_cols = df.select_dtypes(include=np.number).columns.tolist()

print("\n--- 2B. Handling Missing Numerical Data (Mean Imputation) ---")
for col in numerical_impute_cols:
    # Only impute if there were missing values initially
    if col in initial_missing_counts.index:
      mean_val = df[col].mean()
      df[col].fillna(mean_val, inplace=True)
      print(f"Filled {initial_missing_counts[col]} missing values in '{col}' with the mean: {mean_val:.2f}")


# C. Unit Conversion (Standardize to Metric and drop redundant columns)
cols_to_drop = [
    'temperature_fahrenheit', 'feels_like_fahrenheit', # Drop Fahrenheit
    'visibility_miles',                              # Drop Miles
    'pressure_in', 'precip_in',                      # Drop Inches
    'wind_mph', 'gust_mph'                           # Drop MPH
]
df.drop(columns=cols_to_drop, inplace=True)
print("\n--- 2C. Unit Conversion: Removed redundant unit columns (Fahrenheit, Miles, Inches, MPH). ---")


# ====================================================================
# STEP 3: TRANSFORMATION (Normalization and Aggregation)
# ====================================================================

# A. Normalization (Min-Max Scaling)
scaler = MinMaxScaler()
# Apply Min-Max scaling to 'temperature_celsius' (range 0 to 1)
df['temperature_celsius_normalized'] = scaler.fit_transform(df[['temperature_celsius']])
print("\n--- 3A. Normalization: 'temperature_celsius' normalized (Min-Max) and saved as 'temperature_celsius_normalized'. ---")

# B. Aggregation & Filtering (Example: Monthly Averages for a specific country/location)
# --- CHANGE 'India' to your desired country for the final report ---
country_filter = 'India'
df_filtered = df[df['country'] == country_filter].copy()

# Extract the month for aggregation
df_filtered['month'] = df_filtered['last_updated'].dt.to_period('M')

# Group by month and calculate the mean of key numerical variables
monthly_average = df_filtered.groupby('month').agg(
    avg_temp_c=('temperature_celsius', 'mean'),
    avg_humidity=('humidity', 'mean'),
    max_wind_kph=('wind_kph', 'max'),
    count=('country', 'size')
).reset_index()

print(f"\n--- 3B. Aggregation: Monthly average temperature for '{country_filter}'. (First 5 rows) ---")
print(monthly_average.head().to_markdown(index=False, numalign="left", stralign="left"))


# ====================================================================
# STEP 4: EVALUATION AND DELIVERABLE
# ====================================================================

# Save the final cleaned and preprocessed dataset
output_file_name = 'GlobalWeatherRepository_cleaned_processed.csv'
df.to_csv(output_file_name, index=False)
files.download(output_file_name) # Automatically downloads the file in Colab
print(f"\n--- 4. Deliverable: Cleaned dataset saved as '{output_file_name}' and downloaded. ---")


# Generate Summary Document Outline
print("\n\n====================================================================")
print("             SUMMARY DOCUMENT OUTLINE: DATA CLEANING REPORT")
print("====================================================================")
print("1. Data Schema (Final Cleaned Dataset):")
df.info()
print("\n2. Key Variables (Post-Processing Descriptive Statistics):")
print("- 'temperature_celsius' (Original):")
print(df['temperature_celsius'].agg(['mean', 'min', 'max']).to_markdown())
print("\n- 'temperature_celsius_normalized' (New Normalized Column):")
print(df['temperature_celsius_normalized'].agg(['mean', 'min', 'max']).to_markdown())
print("\n- Categorical Variable Counts (Top 5 Weather Conditions):")
print(df['condition_text'].value_counts().head().to_markdown())
print("\n3. Data Quality Issues Handled:")
print(f"- Missing Values: Filled all missing numerical entries (if any) with the column mean. Columns imputed: {list(initial_missing_counts.index.values) or 'None'}")
print("- Unit Inconsistency: Removed redundant columns (Fahrenheit, Miles, Inches, MPH) to standardize on Metric.")
print("- Data Types: Converted 'last_updated' and all time columns to proper datetime objects.")
