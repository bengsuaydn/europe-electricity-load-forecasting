import pandas as pd
import numpy as np

#   =============================================================================
#   Europe Electricity Load Dataset - Data Cleaning Pipeline
#   Task: Group 1 (Data Cleaning)
#
#   Description:
#   This script processes hourly electricity load data (>2M rows). 
#   It performs the following key operations to prepare the data for modeling:
#     1. Checks the cleaned version of scv data to verify cleaning process.
#   =============================================================================

# This code contains file paths that need to be edited according to your system before execution.
# e.g. "C:\ENTER_DATA_ADRESS\DATA.csv"

# 1. Load the Data
file_path = r"C:\ENTER_DATA_ADRESS\DATA_cleaned.csv" 
print("Loading dataset... This might take a few moments.")
df_cleaned = pd.read_csv(file_path)

# 2. Convert DateUTC to Datetime and Sort
print("Formatting dates and sorting data by Country and Time...")
df_cleaned['DateUTC'] = pd.to_datetime(df_cleaned['DateUTC'])
df_cleaned = df_cleaned.sort_values(by=['CountryCode', 'DateUTC']).reset_index(drop=False)
df_cleaned.rename(columns={'index': 'Original_Index'}, inplace=True)

# ---> NEW: Save the original values before ANY modification to evaluate later
df_cleaned['Original_Value'] = df_cleaned['Value'].copy()

# 3. Handle the "0.0" sensor glitches FIRST
print("Replacing 0.0 values with NaN...")
df_cleaned['Value'] = df_cleaned['Value'].replace(0.0, np.nan)

# 4. Anomaly Detection using Rolling Median
print("Calculating rolling median to detect true anomalies...")
rolling_median = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.rolling(window=5, center=True, min_periods=1).median()
)

df_cleaned['Deviation'] = (df_cleaned['Value'] - rolling_median) / rolling_median
threshold = 0.95 

condition = df_cleaned['Deviation'].abs() > threshold
anomalies = df_cleaned[condition]
print(f"\nTotal suspicious jumps found: {len(anomalies)}")

print("Marking extreme anomalies as NaN...")
df_cleaned.loc[condition, 'Value'] = np.nan

# 5. Interpolate the missing values
print("Interpolating NaN values to fix gaps...")
df_cleaned['Value'] = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.interpolate(method='linear')
)

print("Handling edge-case missing values...")
df_cleaned['Value'] = df_cleaned.groupby('CountryCode')['Value'].transform(
    lambda x: x.bfill().ffill()
)

df_cleaned = df_cleaned.drop(columns=['Deviation'])


# ---> NEW: EVALUATION PHASE
print("\n" + "="*50)
print("--- EVALUATION: CHANGED VALUES ---")
print("="*50)

# Compare Original_Value with the new interpolated Value
# We use fillna(-9999) to safely compare NaN values if they exist natively in the raw data
changed_mask = df_cleaned['Original_Value'].fillna(-9999) != df_cleaned['Value'].fillna(-9999)
changed_rows = df_cleaned[changed_mask]

print(f"Total rows modified (0.0s and Anomalies replaced): {len(changed_rows)}\n")

# Display the changes clearly side-by-side
evaluation_df = changed_rows[['Original_Index', 'CountryCode', 'DateUTC', 'Original_Value', 'Value']]
evaluation_df = evaluation_df.rename(columns={'Value': 'New_Interpolated_Value'})

# Set pandas display options to show more rows during evaluation
pd.set_option('display.max_rows', 100)
print(evaluation_df.head(100)) # Shows the first 100 changed rows