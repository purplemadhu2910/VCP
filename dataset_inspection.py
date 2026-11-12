
# Import required libraries
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("tc_dataset.csv")

# Display first 5 records
print("FIRST 5 RECORDS:")
print(df.head())

# Display last 5 records
print("\nLAST 5 RECORDS:")
print(df.tail())

# Display dataset shape
print("\nDATASET SHAPE:")
print(df.shape)

# Display column names
print("\nCOLUMN NAMES:")
print(df.columns.tolist())

# Display dataset information
print("\nDATASET INFORMATION:")
df.info()

# Statistical summary
print("\nSTATISTICAL SUMMARY:")
print(df.describe())

# Check missing values
print("\nMISSING VALUES:")
print(df.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS:")
print(df.duplicated().sum())
#--------------------------------------------------------------------
# PHASE 2: DATA QUALITY CHECKS
print("\n========== DATA QUALITY CHECKS ==========")

# 1. Check unique vehicle categories
print("\nVehicle Categories:")
print(df["Vehicle_Category"].value_counts())

# 2. Check unique UOM values
print("\nVehicle UOM:")
print(df["Vehicle_UOM"].value_counts())

# 3. Check unique ODC values
print("\nODC Values:")
print(df["ODC_Flag"].value_counts())

# 4. Check invalid numerical values
numeric_columns = [
    "Vehicle_Length",
    "Vehicle_Height",
    "Vehicle_Width",
    "Vehicle_Capacity",
    "Vehicle_Transit_Days",
    "Distance_Km",
    "Fuel_Price",
    "Vehicle_Unit_Quantity",
    "Vehicle_Unit_Rate"
]

print("\nZero or Negative Values:")
for col in numeric_columns:
    print(col, ":", (df[col] <= 0).sum())

# 5. Check whether source and destination are same
same_location = (
    df["From_Location"] == df["To_Location"]
).sum()

print("\nSame Source and Destination:", same_location)

# 6. Check detention days against transit days
invalid_detention = df[
    df["Vehicle_Detention_Days"] >
    df["Vehicle_Transit_Days"]
]

print("\nDetention Days Greater Than Transit Days:")
print(invalid_detention[
    ["Vehicle_Transit_Days", "Vehicle_Detention_Days"]
])

# 7. Check freight calculation
expected_freight = (
    df["Vehicle_Unit_Rate"] *
    df["Vehicle_Unit_Quantity"]
)

print("\nFreight Calculation Mismatches:")
print(
    (~np.isclose(
        df["Vehicle_Freight_Charge"],
        expected_freight,
        rtol=1e-4,
        atol=0.02
    )).sum()
)

# 8. Check final amount calculation
expected_total = (
    df["Vehicle_Freight_Charge"] +
    df["Vehicle_Detention_Charge"]
)

print("\nFinal Amount Mismatches:")
print(
    (~np.isclose(
        df["Vehicle_Final_Total_Amount"],
        expected_total,
        rtol=1e-4,
        atol=0.02
    )).sum()
)

# 9. Convert Created_Date to datetime
df["Created_Date"] = pd.to_datetime(
    df["Created_Date"],
    errors="coerce"
)

print("\nCreated_Date Data Type:")
print(df["Created_Date"].dtype)

print("\nInvalid Dates:")
print(df["Created_Date"].isna().sum())

#--------------------------------------------------------------------
# PHASE 3: EXPLORATORY DATA ANALYSIS (EDA) & COST DRIVERS
print("\n========== PHASE 3: COST DRIVER ANALYSIS ==========")

# 1. Overall Cost Summary
print("\nTarget Cost Summary (Final Total Amount):")
print(df["Vehicle_Final_Total_Amount"].describe())

# 2. Average Cost by Vehicle Category
print("\nAverage Total Cost by Vehicle Category:")
category_cost = df.groupby("Vehicle_Category")["Vehicle_Final_Total_Amount"].agg(["mean", "min", "max", "count"]).round(2)
print(category_cost.sort_values(by="mean", ascending=False))

# 3. Vendor Pricing Comparison
print("\nAverage Total Cost by Vendor:")
vendor_cost = df.groupby("Vendor_Name")["Vehicle_Final_Total_Amount"].agg(["mean", "count"]).round(2)
print(vendor_cost.sort_values(by="mean", ascending=False))

# 4. Impact of ODC (Over Dimensional Cargo)
print("\nAverage Cost: ODC vs Non-ODC:")
print(df.groupby("ODC_Flag")["Vehicle_Final_Total_Amount"].mean().round(2))

# 5. Average Detention Cost per Detention Day
active_detention = df[df["Vehicle_Detention_Days"] > 0]
cost_per_detention_day = (active_detention["Vehicle_Detention_Charge"] / active_detention["Vehicle_Detention_Days"]).mean()
print(f"\nAverage Detention Charge per Day: Rs. {cost_per_detention_day:.2f}")

# 6. Correlation with Final Cost
numeric_features = [
    "Distance_Km", "Vehicle_Capacity", "Vehicle_Transit_Days",
    "Fuel_Price", "Vehicle_Freight_Charge", "Vehicle_Final_Total_Amount"
]
print("\nCorrelation with Vehicle_Final_Total_Amount:")
print(df[numeric_features].corr()["Vehicle_Final_Total_Amount"].sort_values(ascending=False).round(4))


#--------------------------------------------------------------------
# PHASE 4: FEATURE ENGINEERING & DATA PREPARATION
print("\n========== PHASE 4: FEATURE ENGINEERING ==========")

# 1. Create derived features
df["Vehicle_Volume"] = df["Vehicle_Length"] * df["Vehicle_Height"] * df["Vehicle_Width"]
df["Created_Month"] = df["Created_Date"].dt.month
df["Created_DayOfWeek"] = df["Created_Date"].dt.dayofweek

# 2. Define Features (X) and Target (y)
# Note: Excluding direct billing math columns to prevent data leakage
feature_cols = [
    "Vehicle_Category", "Vehicle_Volume", "Vehicle_Capacity",
    "ODC_Flag", "Vehicle_Transit_Days", "Distance_Km",
    "Fuel_Price", "Vehicle_UOM", "Vendor_Name", "Created_Month"
]
target_col = "Vehicle_Final_Total_Amount"

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df[target_col]

print(f"Features shape after encoding: {X.shape}")
print(f"Target shape: {y.shape}")

#--------------------------------------------------------------------
# PHASE 5: MODEL TRAINING & EVALUATION
print("\n========== PHASE 5: MODEL TRAINING & BENCHMARKING ==========")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 80-20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        "Model": name,
        "R2 Score": round(r2, 4),
        "MAE (Rs.)": round(mae, 2),
        "RMSE (Rs.)": round(rmse, 2)
    })

results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False)
print("\nModel Benchmark Results:")
print(results_df.to_string(index=False))

# Feature importance from best ensemble model
best_rf = models["Random Forest"]
importances = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(7)
print("\nTop 7 Most Important Predictor Features (Random Forest):")
print(importances.round(4))