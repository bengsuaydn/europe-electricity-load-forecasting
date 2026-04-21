import pandas as pd
import os

print("Starting Feature Engineering...")

# 1. Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(current_dir, '../../data/MHLV_2019_2025_cleaned.csv')
output_path = os.path.join(current_dir, '../../data/engineered_data.csv')

# 2. Load the data (using semicolon)
print(f"Loading data from: {input_path}")
df = pd.read_csv(input_path, sep=';', low_memory=False)

# --- FIX FOR EUROPEAN NUMBERS ---
print("Fixing messy numbers in the Value column...")
df['Value'] = df['Value'].astype(str).str.replace(r'\.(?=.*\.)', '', regex=True).astype(float)

# 3. Convert DateUTC to datetime and sort by time
df['DateUTC'] = pd.to_datetime(df['DateUTC'], format='%d.%m.%Y %H:%M')
df = df.sort_values(by='DateUTC').reset_index(drop=True)

# --- 4. TIME-BASED FEATURES ---
print("Extracting time features...")
df['Hour'] = df['DateUTC'].dt.hour
df['DayOfWeek'] = df['DateUTC'].dt.dayofweek # Monday=0, Sunday=6
df['Month'] = df['DateUTC'].dt.month
df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# --- 5. LAG FEATURES (Grouped by Country) ---
print("Creating lag features...")
df['Load_Lag_1h'] = df.groupby('CountryCode')['Value'].shift(1)
df['Load_Lag_24h'] = df.groupby('CountryCode')['Value'].shift(24)

# --- 6. ROLLING WINDOW FEATURES (Grouped by Country) ---
print("Calculating rolling averages...")
df['Rolling_Mean_24h'] = df.groupby('CountryCode')['Value'].transform(lambda x: x.rolling(window=24, min_periods=1).mean())

# 7. Clean up and save
df = df.dropna()

print(f"Saving engineered data to: {output_path}")
df.to_csv(output_path, sep=';', index=False)

print("✅ Success! Feature engineering complete.")