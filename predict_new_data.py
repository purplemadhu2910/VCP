"""
STAGE 10: PREDICTION ON NEW INPUT DATA
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np
import joblib

print("=== STAGE 10: INFERENCE ON NEW INPUT DATA START ===\n")

# 1. Load the Champion Model and Training Schema
CHAMPION_MODEL_PATH = "champion_model.pkl"
TRAIN_SCHEMA_PATH = "X_train.csv"

model = joblib.load(CHAMPION_MODEL_PATH)
training_cols = pd.read_csv(TRAIN_SCHEMA_PATH, nrows=1).columns.tolist()

print(f"✓ Loaded Champion Model: {type(model).__name__}")
print(f"✓ Training Schema loaded: Expecting {len(training_cols)} features.")

# 2. Define New Raw Trip Inquiries (Simulating real-world L&T requests)
new_trips = [
    {
        "Trip_ID": "TRIP-2026-001 (Standard Full Truckload)",
        "Vehicle_Category": "Truck",
        "Vehicle_Length": 32.0,
        "Vehicle_Height": 8.0,
        "Vehicle_Width": 8.0,
        "Vehicle_Capacity": 25.0,
        "ODC_Flag": "Non-ODC",
        "Vehicle_Transit_Days": 5,
        "Distance_Km": 1450.0,
        "Created_Date": "2026-11-15",
        "Fuel_Price": 88.50,
        "Vehicle_UOM": "MT",
        "Vehicle_Unit_Quantity": 18.0,
        "Vendor_Name": "TCI Freight"
    },
    {
        "Trip_ID": "TRIP-2026-002 (Heavy ODC Machinery Transport)",
        "Vehicle_Category": "SLBT",
        "Vehicle_Length": 25.0,
        "Vehicle_Height": 9.0,
        "Vehicle_Width": 8.5,
        "Vehicle_Capacity": 12.0,
        "ODC_Flag": "ODC",
        "Vehicle_Transit_Days": 3,
        "Distance_Km": 480.0,
        "Created_Date": "2026-11-20",
        "Fuel_Price": 89.20,
        "Vehicle_UOM": "Nos",
        "Vehicle_Unit_Quantity": 2.0,
        "Vendor_Name": "VRL Logistics"
    }
]

df_new = pd.DataFrame(new_trips)

# 3. Preprocessing & Feature Engineering Function
def preprocess_new_data(raw_df, expected_cols):
    df_proc = raw_df.copy()
    
    # Temporal parsing
    df_proc["Created_Date"] = pd.to_datetime(df_proc["Created_Date"])
    df_proc["Created_Month"] = df_proc["Created_Date"].dt.month
    
    # Derived physical & operational features
    df_proc["Vehicle_Volume"] = (df_proc["Vehicle_Length"] * df_proc["Vehicle_Height"] * df_proc["Vehicle_Width"]).round(2)
    df_proc["Capacity_per_Volume"] = (df_proc["Vehicle_Capacity"] / df_proc["Vehicle_Volume"]).round(4)
    df_proc["Km_Per_Day"] = (df_proc["Distance_Km"] / df_proc["Vehicle_Transit_Days"]).round(2)
    
    # One-hot encode categorical features
    categorical_cols = ["Vehicle_Category", "ODC_Flag", "Vehicle_UOM", "Vendor_Name"]
    df_encoded = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=False, dtype=int)
    
    # Align columns with exact training features
    # If a category was not present in this new batch, fill with 0
    # If an extra category was present, it gets dropped safely
    aligned_df = pd.DataFrame(0, index=df_encoded.index, columns=expected_cols)
    for col in expected_cols:
        if col in df_encoded.columns:
            aligned_df[col] = df_encoded[col]
            
    return aligned_df

# 4. Transform New Data
X_new = preprocess_new_data(df_new, training_cols)

# 5. Make Predictions Using Champion Model
predicted_rates = model.predict(X_new)

# 6. Display Business Results
df_new["Predicted_Unit_Rate (Rs.)"] = np.round(predicted_rates, 2)
df_new["Estimated_Freight_Total (Rs.)"] = np.round(
    df_new["Predicted_Unit_Rate (Rs.)"] * df_new["Vehicle_Unit_Quantity"], 2
)

print("--- PREDICTION RESULTS FOR NEW TRIPS ---")
output_columns = [
    "Trip_ID", "Vehicle_Category", "ODC_Flag", "Distance_Km", 
    "Vehicle_Unit_Quantity", "Vehicle_UOM", "Predicted_Unit_Rate (Rs.)", "Estimated_Freight_Total (Rs.)"
]
print(df_new[output_columns].to_string(index=False))

# 7. Save Predictions
df_new.to_csv("new_trips_predictions.csv", index=False)
print("\n✓ Predictions saved to: 'new_trips_predictions.csv'")
print("=== STAGE 10: INFERENCE COMPLETED SUCCESSFULLY ===")