"""
STAGE 1: DATA CLEANING
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np

# 1. Load the raw dataset
RAW_DATA_PATH = "tc_dataset.csv"
CLEANED_DATA_PATH = "cleaned_tc_dataset.csv"

df = pd.read_csv(RAW_DATA_PATH)
print("=== STAGE 1: DATA CLEANING START ===")
print(f"Initial raw dataset shape: {df.shape}")

# 2. Check and remove duplicate rows
initial_rows = len(df)
df = df.drop_duplicates()
duplicates_removed = initial_rows - len(df)
print(f"Duplicate rows removed: {duplicates_removed}")

# 3. Standardize column names (strip any accidental trailing whitespaces)
df.columns = df.columns.str.strip()

# 4. Standardize text columns (strip spaces from string/object columns)
text_columns = df.select_dtypes(include=["object"]).columns
for col in text_columns:
    df[col] = df[col].astype(str).str.strip()

# 5. Convert Created_Date to datetime format
df["Created_Date"] = pd.to_datetime(df["Created_Date"], errors="coerce")
invalid_dates_count = df["Created_Date"].isna().sum()
print(f"Invalid dates detected (NaT): {invalid_dates_count}")

# 6. Verify Critical Numerical Fields (Must be strictly > 0 for valid trips)
# For our model, a trip must have positive distance, capacity, and unit rate
critical_positive_cols = [
    "Vehicle_Length",
    "Vehicle_Height",
    "Vehicle_Width",
    "Vehicle_Capacity",
    "Vehicle_Transit_Days",
    "Distance_Km",
    "Fuel_Price",
    "Vehicle_Unit_Quantity",
    "Vehicle_Unit_Rate"  # Target variable check
]

print("\n--- Non-Positive Value Checks (<= 0) ---")
anomalies_found = False
for col in critical_positive_cols:
    invalid_mask = df[col] <= 0
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        anomalies_found = True
        print(f"⚠️ Column '{col}' has {invalid_count} zero/negative values!")

if not anomalies_found:
    print("All critical numerical features contain valid positive numbers (> 0).")

# 7. Check for same source and destination trips
same_location_mask = df["From_Location"] == df["To_Location"]
same_loc_count = same_location_mask.sum()
print(f"\nTrips where From_Location == To_Location: {same_loc_count}")
if same_loc_count > 0:
    print("Sample records with same source and destination:")
    print(df[same_location_mask][["From_Location", "To_Location", "Distance_Km", "Vehicle_Unit_Rate"]].head())

# 8. Check Missing Values across all columns
missing_counts = df.isnull().sum()
missing_present = missing_counts[missing_counts > 0]
print("\n--- Missing Values Summary ---")
if missing_present.empty:
    print("Zero missing values found in the dataset.")
else:
    print(missing_present)

# 9. Save Cleaned Dataset (Original CSV remains completely untouched)
df.to_csv(CLEANED_DATA_PATH, index=False)
print(f"\nCleaned dataset saved as: '{CLEANED_DATA_PATH}'")
print(f"Final cleaned dataset shape: {df.shape}")
print("=== STAGE 1: DATA CLEANING COMPLETED ===")