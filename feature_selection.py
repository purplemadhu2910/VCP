"""
STAGE 4: FEATURE SELECTION
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np

# 1. Load engineered dataset
INPUT_PATH = "engineered_tc_dataset.csv"
OUTPUT_PATH = "selected_features_tc_dataset.csv"

df = pd.read_csv(INPUT_PATH)
print("=== STAGE 4: FEATURE SELECTION START ===")
print(f"Total columns available: {df.shape[1]}")

# 2. Define Explicit Feature Groups
TARGET_COL = "Vehicle_Unit_Rate"

NUMERICAL_FEATURES = [
    "Distance_Km",
    "Vehicle_Capacity",
    "Vehicle_Volume",
    "Capacity_per_Volume",
    "Km_Per_Day",
    "Vehicle_Unit_Quantity",
    "Fuel_Price",
    "Created_Month"
]

CATEGORICAL_FEATURES = [
    "Vehicle_Category",
    "ODC_Flag",
    "Vehicle_UOM",
    "Vendor_Name"
]

LEAKAGE_COLUMNS = [
    "Vehicle_Freight_Charge",
    "Vehicle_Detention_Charge",
    "Vehicle_Final_Total_Amount",
    "Vehicle_Detention_Days"
]

# 3. Confirm Zero Leakage Columns in Feature Set
assert not any(col in NUMERICAL_FEATURES + CATEGORICAL_FEATURES for col in LEAKAGE_COLUMNS), \
    "ERROR: Data leakage detected in selected features!"

print("\n✓ Data Leakage Check Passed: No billing or post-transit columns included.")

# 4. Construct Final Feature Matrix and Target
selected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
df_selected = df[selected_cols].copy()

print(f"\nSelected Feature Summary:")
print(f"- Numerical features ({len(NUMERICAL_FEATURES)}): {NUMERICAL_FEATURES}")
print(f"- Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")
print(f"- Target variable (1): '{TARGET_COL}'")
print(f"- Total retained columns: {df_selected.shape[1]}")

# 5. Correlation Check Among Selected Numerical Features
print("\n--- Correlation Matrix of Selected Numerical Features ---")
num_corr = df_selected[NUMERICAL_FEATURES + [TARGET_COL]].corr()
print(num_corr[TARGET_COL].sort_values(ascending=False).round(4))

# 6. Save the Selected Feature Dataset
df_selected.to_csv(OUTPUT_PATH, index=False)
print(f"\nFiltered dataset saved successfully as: '{OUTPUT_PATH}'")
print(f"Final selected dataset shape: {df_selected.shape}")
print("=== STAGE 4: FEATURE SELECTION COMPLETED ===")