"""
STAGE 3: FEATURE ENGINEERING
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np

# 1. Load cleaned data
CLEANED_DATA_PATH = "cleaned_tc_dataset.csv"
ENGINEERED_DATA_PATH = "engineered_tc_dataset.csv"

df = pd.read_csv(CLEANED_DATA_PATH)
df["Created_Date"] = pd.to_datetime(df["Created_Date"])

print("=== STAGE 3: FEATURE ENGINEERING START ===")
print(f"Dataset shape before engineering: {df.shape}")

# -------------------------------------------------------------
# 1. DERIVED PHYSICAL & OPERATIONAL FEATURES
# -------------------------------------------------------------
# Vehicle 3D volume (cubic dimensions)
df["Vehicle_Volume"] = (df["Vehicle_Length"] * df["Vehicle_Height"] * df["Vehicle_Width"]).round(2)

# Vehicle cargo density capability (Tons per unit volume)
df["Capacity_per_Volume"] = (df["Vehicle_Capacity"] / df["Vehicle_Volume"]).round(4)

# Transit intensity / average daily coverage
df["Km_Per_Day"] = (df["Distance_Km"] / df["Vehicle_Transit_Days"]).round(2)

# Lane / Corridor Identifier
df["Route"] = df["From_Location"] + " -> " + df["To_Location"]

# -------------------------------------------------------------
# 2. TEMPORAL FEATURES (From Created_Date)
# -------------------------------------------------------------
df["Created_Month"] = df["Created_Date"].dt.month
df["Created_Quarter"] = df["Created_Date"].dt.quarter
df["Created_DayOfWeek"] = df["Created_Date"].dt.dayofweek

# -------------------------------------------------------------
# 3. VERIFY DATA QUALITY OF NEW FEATURES
# -------------------------------------------------------------
# Check for any infinite values or NaNs that could result from division
engineered_cols = [
    "Vehicle_Volume", "Capacity_per_Volume", "Km_Per_Day", 
    "Route", "Created_Month", "Created_Quarter", "Created_DayOfWeek"
]

print("\n--- Summary of Newly Engineered Features ---")
print(df[engineered_cols].describe(include="all").T[["count", "mean", "std", "min", "max"]])

# Check correlation of new numerical features with Target (Vehicle_Unit_Rate)
new_numeric_cols = ["Vehicle_Volume", "Capacity_per_Volume", "Km_Per_Day", "Created_Month", "Vehicle_Unit_Rate"]
print("\n--- Correlation of Engineered Features with Target (Vehicle_Unit_Rate) ---")
print(df[new_numeric_cols].corr()["Vehicle_Unit_Rate"].sort_values(ascending=False).round(4))

# -------------------------------------------------------------
# 4. SAVE ENGINEERED DATASET
# -------------------------------------------------------------
df.to_csv(ENGINEERED_DATA_PATH, index=False)
print(f"\nEngineered dataset saved successfully as: '{ENGINEERED_DATA_PATH}'")
print(f"Final engineered dataset shape: {df.shape}")
print("=== STAGE 3: FEATURE ENGINEERING COMPLETED ===")