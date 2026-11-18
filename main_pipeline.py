"""
=============================================================================
END-TO-END MACHINE LEARNING PIPELINE: VEHICLE TRANSPORTATION UNIT RATE PREDICTION
Author: Machine Learning Intern
Dataset: 10,000 Transportation Records across India
Target Variable (Y): Vehicle_Unit_Rate
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

print("=" * 70)
print("VEHICLE TRANSPORTATION UNIT RATE PREDICTION - END-TO-END PIPELINE")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING & DATA CLEANING
# =============================================================================
print("\n[STAGE 1] Loading and Cleaning Data...")
RAW_PATH = "tc_dataset.csv"
df = pd.read_csv(RAW_PATH)
print(f"-> Raw dataset loaded: {df.shape}")

# Deduplication & string whitespace stripping
df = df.drop_duplicates()
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str).str.strip()

# Datetime conversion
df["Created_Date"] = pd.to_datetime(df["Created_Date"], errors="coerce")
df = df.dropna(subset=["Created_Date"])

# Positive values filtering
critical_positive = [
    "Vehicle_Length", "Vehicle_Height", "Vehicle_Width",
    "Vehicle_Capacity", "Vehicle_Transit_Days", "Distance_Km",
    "Fuel_Price", "Vehicle_Unit_Quantity", "Vehicle_Unit_Rate"
]
for col in critical_positive:
    df = df[df[col] > 0]

print(f"-> Cleaned dataset shape: {df.shape}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[STAGE 2] Engineering Logistics Features...")
df["Vehicle_Volume"] = (df["Vehicle_Length"] * df["Vehicle_Height"] * df["Vehicle_Width"]).round(2)
df["Capacity_per_Volume"] = (df["Vehicle_Capacity"] / df["Vehicle_Volume"]).round(4)
df["Km_Per_Day"] = (df["Distance_Km"] / df["Vehicle_Transit_Days"]).round(2)
df["Created_Month"] = df["Created_Date"].dt.month

# =============================================================================
# 3. FEATURE SELECTION (STRICT DATA LEAKAGE PREVENTION)
# =============================================================================
print("\n[STAGE 3] Selecting Features & Eliminating Data Leakage...")
# NOTE: Vehicle_Freight_Charge, Vehicle_Detention_Charge, Vehicle_Final_Total_Amount
# and Vehicle_Detention_Days are strictly excluded because they are downstream billing outputs.
TARGET_COL = "Vehicle_Unit_Rate"
NUMERICAL_COLS = [
    "Distance_Km", "Vehicle_Capacity", "Vehicle_Volume",
    "Capacity_per_Volume", "Km_Per_Day", "Vehicle_Unit_Quantity",
    "Fuel_Price", "Created_Month"
]
CATEGORICAL_COLS = ["Vehicle_Category", "ODC_Flag", "Vehicle_UOM", "Vendor_Name"]

df_selected = df[NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL]].copy()
print(f"-> Selected {len(NUMERICAL_COLS)} numerical and {len(CATEGORICAL_COLS)} categorical features.")

# =============================================================================
# 4. CATEGORICAL ENCODING
# =============================================================================
print("\n[STAGE 4] One-Hot Encoding Categorical Variables...")
df_encoded = pd.get_dummies(df_selected, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)
print(f"-> Encoded dataset shape: {df_encoded.shape} (23 features + 1 target)")

# =============================================================================
# 5. TRAIN-TEST SPLIT (80-20 HOLDOUT)
# =============================================================================
print("\n[STAGE 5] Creating 80-20 Train-Test Split (Reproducible)...")
X = df_encoded.drop(columns=[TARGET_COL])
y = df_encoded[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"-> Training Set: {X_train.shape} (80%)")
print(f"-> Testing Holdout: {X_test.shape} (20%)")

# =============================================================================
# 6. MODEL TRAINING & BENCHMARKING
# =============================================================================
print("\n[STAGE 6] Training 4 Regression Models...")
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0, random_state=42))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():
    print(f"   Fitting {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Train & Test evaluation
    train_r2 = model.score(X_train, y_train)
    y_pred = model.predict(X_test)
    
    test_r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    results.append({
        "Model": name,
        "Train R2": round(train_r2, 4),
        "Test R2": round(test_r2, 4),
        "Gen. Gap": round(train_r2 - test_r2, 4),
        "MAE (Rs.)": round(mae, 2),
        "RMSE (Rs.)": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    })

# =============================================================================
# 7. MODEL EVALUATION & CHAMPION SELECTION
# =============================================================================
print("\n[STAGE 7] Model Benchmark Results (Evaluated on 2,000 Unseen Test Trips):")
results_df = pd.DataFrame(results).sort_values(by="Test R2", ascending=False)
print(results_df.to_string(index=False))

champion_name = results_df.iloc[0]["Model"]
champion_model = trained_models[champion_name]
print(f"\n🏆 CHAMPION MODEL: {champion_name} (Test R2: {results_df.iloc[0]['Test R2']})")

# Save the Champion Model
joblib.dump(champion_model, "champion_model.pkl")
print("-> Champion model saved as 'champion_model.pkl'")

# =============================================================================
# 8. TOP COST DRIVERS (FEATURE IMPORTANCE)
# =============================================================================
if hasattr(champion_model, "feature_importances_"):
    print("\n[STAGE 8] Top 5 Key Cost Drivers:")
    importances = pd.Series(champion_model.feature_importances_, index=X_train.columns)
    top_5 = importances.sort_values(ascending=False).head(5)
    for rank, (feat, score) in enumerate(top_5.items(), 1):
        print(f"   {rank}. {feat:<28}: {score*100:.1f}%")

# =============================================================================
# 9. SAMPLE PREDICTION DEMONSTRATION
# =============================================================================
print("\n[STAGE 9] Live Rate Prediction Demonstration:")
sample_trip = pd.DataFrame(0, index=[0], columns=X_train.columns)
sample_trip["Distance_Km"] = 1450.0
sample_trip["Vehicle_Capacity"] = 25.0
sample_trip["Vehicle_Volume"] = 2048.0
sample_trip["Capacity_per_Volume"] = 0.0122
sample_trip["Km_Per_Day"] = 290.0
sample_trip["Vehicle_Unit_Quantity"] = 18.0
sample_trip["Fuel_Price"] = 89.5
sample_trip["Created_Month"] = 11
sample_trip["Vehicle_Category_Truck"] = 1
sample_trip["Vendor_Name_TCI Freight"] = 1

pred_rate = champion_model.predict(sample_trip)[0]
est_total = pred_rate * 18.0
print(f"   Trip Specs: Truck (32x8x8 ft), 1450 Km, 18 MT Quantity, TCI Freight")
print(f"   -> Predicted Unit Rate: Rs. {pred_rate:,.2f} / MT")
print(f"   -> Estimated Total Freight: Rs. {est_total:,.2f}")
print("=" * 70)
print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 70)
