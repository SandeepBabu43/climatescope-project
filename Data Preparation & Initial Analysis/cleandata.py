import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from google.colab import files
import io

# ====================================================================
# A. FUNCTION DEFINITIONS
# ====================================================================

def clean_data_types(df):
    """Converts date/time columns to proper datetime objects."""
    print("-> Cleaning Data Types...")
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    # Handle time-only columns (moon/sun times)
    time_cols = ['sunrise', 'sunset', 'moonrise', 'moonset']
    for col in time_cols:
        # 'coerce' turns invalid entries into NaT (Not a Time)
        df[col] = pd.to_datetime(df[col], format='%I:%M %p', errors='coerce')
    return df

def standardize_units(df):
    """Removes redundant unit columns to standardize on Metric (C, kph, mb)."""
    print("-> Standardizing Units (Removing redundant columns)...")
    cols_to_drop = [
        'temperature_fahrenheit', 'feels_like_fahrenheit', 
        'visibility_miles', 'pressure_in', 'precip_in', 
        'wind_mph', 'gust_mph', 'last_updated_epoch'
    ]
    # Use errors='ignore' in case some columns were dropped in a previous run
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    return df

def handle_missing_values(df):
    """Imputes missing numerical values with the column mean."""
    initial_missing_counts = df.isnull().sum()
    numerical_impute_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    print("-> Handling Missing Numerical Data (Mean Imputation)...")
    imputed_cols = []
    for col in numerical_impute_cols:
        if initial_missing_counts[col] > 0:
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)
            imputed_cols.append(f"'{col}' ({initial_missing_counts[col]} imputed)")
            
    print(f"   Columns Imputed: {', '.join(imputed_cols) or 'None'}")
    return df

def normalize_features(df, column='temperature_celsius'):
    """Applies Min-Max scaling to a specified column."""
    print(f"-> Normalizing '{column}' using Min-Max Scaling...")
    scaler = MinMaxScaler()
    new_col_name = f'{column}_normalized'
    df[new_col_name] = scaler.fit_transform(df[[column]])
    return df

def aggregate_data(df, country_filter='India'):
    """Filters data by country and aggregates to monthly averages."""
    print(f"-> Aggregating Data for '{country_filter}' to Monthly Averages...")
    
    # 1. Filter
    df_filtered = df[df['country'] == country_filter].copy()
    
    # Check if the datetime column exists (from clean_data_types)
    if 'last_updated' not in df_filtered.columns:
        df_filtered['last_updated'] = pd.to_datetime(df_filtered['last_updated'])

    df_filtered['month'] = df_filtered['last_updated'].dt.to_period('M')

    # 2. Aggregate
    monthly_avg = df_filtered.groupby('month').agg(
        avg_temp_c=('temperature_celsius', 'mean'),
        avg_humidity=('humidity', 'mean'),
        max_wind_kph=('wind_kph', 'max'),
        count=('country', 'size')
    ).reset_index()
    
    return monthly_avg

def generate_summary(df, monthly_avg):
    """Prints the final summary report/outline."""
    print("\n\n====================================================================")
    print("             SUMMARY DOCUMENT OUTLINE: DATA CLEANING REPORT")
    print("====================================================================")
    
    print("1. Data Schema (Final Cleaned Dataset):")
    df.info()
    
    print("\n2. Key Variables (Post-Processing Descriptive Statistics):")
    print("- 'temperature_celsius' (Original): Mean, Min, Max:")
    print(df['temperature_celsius'].agg(['mean', 'min', 'max']).to_markdown())
    
    if 'temperature_celsius_normalized' in df.columns:
        print("\n- 'temperature_celsius_normalized' (New Normalized Column):")
        print(df['temperature_celsius_normalized'].agg(['mean', 'min', 'max']).to_markdown())
    
    print("\n3. Data Quality Issues Handled (via functions):")
    print("- Data Types: Handled by `clean_data_types()`.")
    print("- Unit Inconsistency: Handled by `standardize_units()`.")
    print("- Missing Values: Handled by `handle_missing_values()`.")
    print("\n4. Aggregation Example (Monthly Averages):")
    print(monthly_avg.head().to_markdown(index=False, numalign="left", stralign="left"))


# ====================================================================
# B. EXECUTION (The main script flow)
# ====================================================================
if __name__ == "__main__":
    # --- 0. FILE LOAD (Colab specific) ---
    print("Please upload the 'GlobalWeatherRepository.csv' file when prompted.")
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0] 
    df = pd.read_csv(io.BytesIO(uploaded[file_path]))
    print(f"\nSuccessfully loaded dataset: {file_path}")
    print("====================================================================")
    
    # --- 1. DATA CLEANING & PREPROCESSING PIPELINE (Function Calls) ---
    df = clean_data_types(df)
    df = standardize_units(df)
    df = handle_missing_values(df)
    df = normalize_features(df)
    
    # --- 2. AGGREGATION ---
    monthly_avg_df = aggregate_data(df, country_filter='India') 
    # Change 'India' to any other country for analysis if desired
    
    # --- 3. EVALUATION & DELIVERABLE ---
    output_file_name = 'GlobalWeatherRepository_cleaned_processed_functional.csv'
    df.to_csv(output_file_name, index=False)
    files.download(output_file_name)
    print(f"\n--- Deliverable: Cleaned dataset saved as '{output_file_name}' and downloaded. ---")
    
    generate_summary(df, monthly_avg_df)
