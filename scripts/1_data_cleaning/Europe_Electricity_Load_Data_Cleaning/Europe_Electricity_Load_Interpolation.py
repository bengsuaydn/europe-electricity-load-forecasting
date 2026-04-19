import pandas as pd
import numpy as np


#   =============================================================================
#   Europe Electricity Load Dataset - Data Cleaning Pipeline
#   Task: Group 1 (Data Cleaning)
#
#   Description:
#   This script processes and cleans hourly electricity load data (more than 2M rows). 
#   It performs the following key operations to prepare the data for modeling:
#     1. Formats datetime objects and removes redundant columns.
#     2. Handles physical sensor failures by replacing exact 0.0 values with NaN.
#     3. Detects anomalous spikes/glitches using a 5-hour rolling median 
#        (60% deviation threshold) to prevent the rebound effect.
#     4. Imputes missing values using linear interpolation (grouped by country) 
#        to maintain the integrity of the time series flow.
#     5. Sorts the final dataset hierarchically (Year -> CountryCode -> DateUTC) 
#        and exports a clean, ready-to-use CSV for Feature Engineering.
#   =============================================================================

# This code contains file paths that need to be edited according to your system before execution.
# e.g. "C:\ENTER_DATA_ADRESS\DATA.csv"

# ==========================================================
# PHASE 1: DATA LOADING & PREPARATION
# ==========================================================
file_path = r"C:\ENTER_DATA_ADRESS\DATA.csv" 
print("Loading dataset... This might take a few moments.")
df_cleaned = pd.read_csv(file_path)

# Convert DateUTC to Datetime and Sort
print("Formatting dates and sorting data by DateUTC and CountryCode...")
df_cleaned['DateUTC'] = pd.to_datetime(df_cleaned['DateUTC'])
df_cleaned = df_cleaned.sort_values(by=['DateUTC', 'CountryCode']).reset_index(drop=True)

# Save the original values for internal evaluation (not for the final CSV)
df_cleaned['Original_Value'] = df_cleaned['Value'].copy()

# ==========================================================
# PHASE 2: ANOMALY DETECTION & DATA CLEANING
# ==========================================================
# 1. Handle the "0.0" sensor glitches
print("Replacing 0.0 values with NaN...")
df_cleaned['Value'] = df_cleaned['Value'].replace(0.0, np.nan)

# 2. Anomaly Detection using Rolling Median
print("Calculating rolling median to detect true anomalies...")
# Note: We group by CountryCode to ensure rolling windows stay within each country
rolling_median = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.rolling(window=5, center=True, min_periods=1).median()
)

df_cleaned['Deviation'] = (df_cleaned['Value'] - rolling_median) / rolling_median
threshold = 0.60 

condition = df_cleaned['Deviation'].abs() > threshold
anomalies = df_cleaned[condition]
print(f"\nTotal suspicious jumps found and marked as NaN: {len(anomalies)}")
df_cleaned.loc[condition, 'Value'] = np.nan

# ==========================================================
# PHASE 3: INTERPOLATION (FILLING THE GAPS)
# ==========================================================
print("Interpolating NaN values to fix gaps...")
df_cleaned['Value'] = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.interpolate(method='linear')
)

print("Handling edge-case missing values...")
df_cleaned['Value'] = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.bfill().ffill()
)

# ==========================================================
# PHASE 4: FINAL SORTING & EXPORTING THE CLEANED DATASET
# ==========================================================
output_file = r"\ENTER_DATA_ADRESS\DATA_cleaned.csv"

# 1. Cleanup: Explicitly dropping all temporary columns
final_columns_to_drop = ['Original_Value', 'Deviation']
df_final = df_cleaned.drop(columns=final_columns_to_drop, errors='ignore')

# 2. Extract Year and Apply FINAL SORTING
# Safely extract the year from DateUTC to ensure accurate year-based sorting
print("Extracting year and applying final sorting (Year -> CountryCode -> DateUTC)...")
df_final['year'] = df_final['DateUTC'].dt.year

# Sort by year, then country, then exact time
df_final = df_final.sort_values(by=['year', 'CountryCode', 'DateUTC']).reset_index(drop=True)

print(f"\nExporting completely cleaned dataset to: {output_file}")
# index=False ensures that the pandas index column is not saved in the CSV
df_final.to_csv(output_file, index=False)

print("\nSUCCESS: Final CSV is ready, perfectly sorted (Year -> CountryCode -> DateUTC), and clean!")
print(f"Final Columns: {list(df_final.columns)}")
print("="*50)