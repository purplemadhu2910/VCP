"""
=============================================================================
TESTING & EVALUATION RUNNER (FOR NEW / UNSEEN EVALUATION DATASETS)
Project: Vehicle Transportation Unit Rate Prediction
Target Variable (Y): Vehicle_Unit_Rate
=============================================================================
Usage:
    python test_on_new_dataset.py [path_to_your_dataset.csv]

Example:
    python test_on_new_dataset.py test_trips.csv
    (If no argument is passed, it looks for 'test_dataset.csv' or uses 'tc_dataset.csv')
=============================================================================
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 70)
print("VEHICLE UNIT RATE PREDICTOR - EVALUATION ON NEW DATASET")
print("=" * 70)

# 1. Determine Input File
if len(sys.argv) > 1:
    INPUT_FILE = sys.argv[1]
else:
    if os.path.exists("test_dataset.csv"):
        INPUT_FILE = "test_dataset.csv"
    else:
        INPUT_FILE = "tc_dataset.csv"

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: File '{INPUT_FILE}' not found. Please provide a valid CSV path.")
    sys.exit(1)

print(f"-> Loading evaluation dataset: '{INPUT_FILE}'")
df_raw = pd.read_csv(INPUT_FILE)
print(f"-> Total records in evaluation file: {len(df_raw)}")

# 2. Load Model and Training Schema
MODEL_PATH = "champion_model.pkl"
SCHEMA_PATH = "X_train.csv"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCHEMA_PATH):
    print("❌ Error: 'champion_model.pkl' or 'X_train.csv' not found. Run main_pipeline.py first.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
expected_cols = pd.read_csv(SCHEMA_PATH, nrows=1).columns.tolist()
print(f"-> Loaded Champion Model: {type(model).__name__}")
print(f"-> Expected model features: {len(expected_cols)}")

# 3. Clean and Preprocess New Dataset
df_clean = df_raw.copy()
df_clean.columns = df_clean.columns.str.strip()

# Fill missing numeric values safely if any
df_clean.fillna(df_clean.median(numeric_only=True), inplace=True)

# Datetime & Seasonality
if "Created_Date" in df_clean.columns:
    df_clean["Created_Date"] = pd.to_datetime(df_clean["Created_Date"], errors="coerce")
    df_clean["Created_Month"] = df_clean["Created_Date"].dt.month.fillna(1).astype(int)
else:
    df_clean["Created_Month"] = 1

# Feature Engineering
df_clean["Vehicle_Volume"] = (df_clean["Vehicle_Length"] * df_clean["Vehicle_Height"] * df_clean["Vehicle_Width"]).round(2)
df_clean["Capacity_per_Volume"] = (df_clean["Vehicle_Capacity"] / df_clean["Vehicle_Volume"]).round(4)
df_clean["Km_Per_Day"] = (df_clean["Distance_Km"] / df_clean["Vehicle_Transit_Days"]).round(2)

# One-hot encoding & Schema alignment
categorical_cols = ["Vehicle_Category", "ODC_Flag", "Vehicle_UOM", "Vendor_Name"]
existing_cats = [col for col in categorical_cols if col in df_clean.columns]
df_encoded = pd.get_dummies(df_clean, columns=existing_cats, dtype=int)

# Align columns with exact training features (Missing categories filled with 0)
X_aligned = pd.DataFrame(0, index=df_clean.index, columns=expected_cols)
for col in expected_cols:
    if col in df_encoded.columns:
        X_aligned[col] = df_encoded[col]

# 4. Generate Predictions
print("\n-> Generating rate predictions using Champion Model...")
predicted_rates = model.predict(X_aligned)
df_raw["Predicted_Unit_Rate (Rs.)"] = np.round(predicted_rates, 2)

if "Vehicle_Unit_Quantity" in df_raw.columns:
    df_raw["Estimated_Freight_Total (Rs.)"] = np.round(
        df_raw["Predicted_Unit_Rate (Rs.)"] * df_raw["Vehicle_Unit_Quantity"], 2
    )

# 5. Model Evaluation (If actual Vehicle_Unit_Rate exists in their dataset)
if "Vehicle_Unit_Rate" in df_raw.columns:
    actual_y = df_raw["Vehicle_Unit_Rate"].values
    r2 = r2_score(actual_y, predicted_rates)
    mae = mean_absolute_error(actual_y, predicted_rates)
    rmse = np.sqrt(mean_squared_error(actual_y, predicted_rates))
    mape = np.mean(np.abs((actual_y - predicted_rates) / actual_y)) * 100

    print("\n" + "=" * 70)
    print("ACCURACY BENCHMARK ON YOUR DATASET:")
    print("=" * 70)
    print(f"  • R² Score:   {r2:.4f} ({r2*100:.2f}%)")
    print(f"  • MAE:        Rs. {mae:,.2f}")
    print(f"  • RMSE:       Rs. {rmse:,.2f}")
    print(f"  • MAPE:       {mape:.2f}%")
    print("=" * 70)

# 6. Save Predictions to CSV
OUTPUT_CSV = "predictions_output.csv"
df_raw.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Predictions successfully saved to: '{OUTPUT_CSV}'")

# Preview First 5 Predictions
preview_cols = ["Vehicle_Category", "Distance_Km", "Predicted_Unit_Rate (Rs.)"]
if "Estimated_Freight_Total (Rs.)" in df_raw.columns:
    preview_cols.append("Estimated_Freight_Total (Rs.)")
if "Vehicle_Unit_Rate" in df_raw.columns:
    preview_cols.insert(2, "Vehicle_Unit_Rate")

print("\n--- Sample Predictions Preview (First 5 records) ---")
print(df_raw[preview_cols].head().to_string(index=False))
print("\n" + "=" * 70)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 70)
