import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
import io

# Set visualization style globally
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 100

# List of saved image files for final download
SAVED_FILES = []

# ====================================================================
# A. FUNCTION DEFINITIONS (Core Analysis & Visualization)
# ====================================================================

def load_and_initial_clean(file_path, uploaded_files):
    """Loads data and performs essential cleaning (type conversion, column drops)."""
    print("-> Loading and performing initial data cleaning...")
    
    # Load data from BytesIO object provided by Colab's files.upload()
    df = pd.read_csv(io.BytesIO(uploaded_files[file_path]))

    # Type Conversion
    df['last_updated'] = pd.to_datetime(df['last_updated'])
    time_cols = ['sunrise', 'sunset', 'moonrise', 'moonset']
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], format='%I:%M %p', errors='coerce')

    # Drop Redundant/Non-Metric Columns
    cols_to_drop = [
        'temperature_fahrenheit', 'feels_like_fahrenheit', 
        'visibility_miles', 'pressure_in', 'precip_in', 
        'wind_mph', 'gust_mph', 'last_updated_epoch'
    ]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # Extract Time Features for Analysis
    df['month'] = df['last_updated'].dt.month
    df['hour'] = df['last_updated'].dt.hour
    df['date'] = df['last_updated'].dt.date
    
    print(f"   Initial Cleaning Complete. Dataset shape: {df.shape}")
    return df

def perform_statistical_analysis(df):
    """Performs descriptive stats, correlation, and seasonal/diurnal trend analysis."""
    print("\n--- 1. Performing Statistical Analysis ---")
    
    # Descriptive Statistics
    core_vars = ['temperature_celsius', 'humidity', 'wind_kph', 'uv_index', 'precip_mm']
    stats_summary = df[core_vars].describe().T
    print("\n   [1.1] Descriptive Statistics (Core Variables):")
    print(stats_summary.to_markdown(floatfmt=".2f"))

    # Correlation Analysis
    numerical_df = df.select_dtypes(include=np.number).copy()
    corr_matrix = numerical_df.corr()
    temp_corr = corr_matrix['temperature_celsius'].sort_values(ascending=False).drop('temperature_celsius')
    print("\n   [1.2] Top 5 Correlations with Temperature:")
    print(temp_corr.head(5).to_markdown(floatfmt=".3f"))
    
    # Seasonal & Daily Trends
    monthly_avg = df.groupby('month')['temperature_celsius'].mean().reset_index()
    hourly_avg = df.groupby('hour')['temperature_celsius'].mean().reset_index()
    print("\n   [1.3] Seasonal Trends (Monthly Avg Temp):")
    print(monthly_avg.to_markdown(floatfmt=".2f", index=False))
    
    return monthly_avg, hourly_avg

def identify_extreme_events(df, quantile=0.99, top_n=5):
    """Identifies and reports the top N extreme heat and wind events."""
    print("\n--- 2. Identifying Extreme Weather Events ---")

    # Define Extreme Thresholds dynamically
    temp_threshold = df['temperature_celsius'].quantile(quantile)
    wind_threshold = df['wind_kph'].quantile(quantile)
    print(f"   Thresholds: Extreme Heat > {temp_threshold:.2f}°C, Extreme Wind > {wind_threshold:.2f} kph")

    # Identify Extreme Heat
    extreme_heat = df[df['temperature_celsius'] > temp_threshold].sort_values(
        'temperature_celsius', ascending=False
    ).head(top_n)

    print(f"\n   [2.1] Top {top_n} Extreme Heat Events:")
    print(extreme_heat[['country', 'location_name', 'date', 'temperature_celsius', 'condition_text']].to_markdown(index=False))

    # Identify Extreme Wind
    extreme_wind = df[df['wind_kph'] > wind_threshold].sort_values(
        'wind_kph', ascending=False
    ).head(top_n)

    print(f"\n   [2.2] Top {top_n} Extreme Wind Events:")
    print(extreme_wind[['country', 'location_name', 'date', 'wind_kph', 'condition_text']].to_markdown(index=False))
    
    return extreme_heat, extreme_wind

def compare_regions(df, top_n=10):
    """Calculates and reports comparative metrics across the top N regions/countries."""
    print("\n--- 3. Regional Comparative Analysis ---")

    # Determine the Top N Countries by observation count
    top_countries = df['country'].value_counts().head(top_n).index.tolist()

    # Calculate average metrics for these top countries
    regional_comparison = df[df['country'].isin(top_countries)].groupby('country').agg(
        Avg_Temp_C=('temperature_celsius', 'mean'),
        Avg_Humidity=('humidity', 'mean'),
        Max_Wind_KPH=('wind_kph', 'max'),
        Observation_Count=('country', 'size')
    ).sort_values('Avg_Temp_C', ascending=False).reset_index()

    print(f"\n   [3.1] Comparative Metrics for Top {top_n} Countries:")
    print(regional_comparison.to_markdown(floatfmt=".2f", index=False))
    
    return regional_comparison

def generate_visualizations(df, monthly_avg_df, regional_comparison_df):
    """Generates, displays, and saves the three key visualizations for the report/dashboard."""
    global SAVED_FILES
    print("\n--- 4. Generating Visualization Deliverables ---")

    # --- 1. Monthly Trend (Line Chart) ---
    filename1 = 'monthly_temp_trend.png'
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=monthly_avg_df, x='month', y='temperature_celsius', marker='o')
    plt.title('Monthly Average Temperature Trend')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.xticks(monthly_avg_df['month'])
    plt.savefig(filename1)
    plt.show() # FIX: Display plot
    SAVED_FILES.append(filename1)
    print(f"   [{filename1}] saved (Seasonal Line Chart).")

    # --- 2. Regional Comparison (Bar Chart) ---
    filename2 = 'country_temp_comparison.png'
    plt.figure(figsize=(12, 6))
    regional_comparison_df = regional_comparison_df.sort_values('Avg_Temp_C', ascending=False)
    sns.barplot(data=regional_comparison_df, x='country', y='Avg_Temp_C', palette='viridis')
    plt.title('Average Temperature Comparison Across Top Countries')
    plt.xlabel('Country')
    plt.ylabel('Average Temperature (°C)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename2)
    plt.show() # FIX: Display plot
    SAVED_FILES.append(filename2)
    print(f"   [{filename2}] saved (Regional Bar Chart).")
    
    # --- 3. Correlation Heatmap ---
    filename3 = 'correlation_heatmap.png'
    heatmap_cols = ['temperature_celsius', 'feels_like_celsius', 'humidity', 'wind_kph', 'cloud', 'uv_index', 'precip_mm']
    heatmap_data = df[heatmap_cols].corr()

    plt.figure(figsize=(8, 7))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, linewidths=.5, linecolor='black')
    plt.title('Correlation Heatmap of Key Weather Variables')
    plt.tight_layout()
    plt.savefig(filename3)
    plt.show() # FIX: Display plot
    SAVED_FILES.append(filename3)
    print(f"   [{filename3}] saved (Correlation Heatmap).")

# ====================================================================
# B. EXECUTION (The main script flow)
# ====================================================================
if __name__ == "__main__":
    
    # --- 0. FILE LOAD (Colab specific) ---
    print("Please upload the 'GlobalWeatherRepository.csv' file when prompted.")
    uploaded = files.upload()
    file_path = list(uploaded.keys())[0] 
    
    # --- 1. DATA PREPARATION ---
    df_cleaned = load_and_initial_clean(file_path, uploaded)
    
    # --- 2. CORE ANALYSIS ---
    monthly_avg, hourly_avg = perform_statistical_analysis(df_cleaned)
    extreme_heat, extreme_wind = identify_extreme_events(df_cleaned)
    regional_comparison = compare_regions(df_cleaned)
    
    # --- 3. VISUALIZATION DELIVERABLE ---
    generate_visualizations(df_cleaned, monthly_avg, regional_comparison)
