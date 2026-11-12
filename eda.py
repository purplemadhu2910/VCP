"""
STAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)

# 1. Load the cleaned dataset
DATA_PATH = "cleaned_tc_dataset.csv"
df = pd.read_csv(DATA_PATH)

print("=== STAGE 2: EXPLORATORY DATA ANALYSIS (EDA) START ===\n")

# -------------------------------------------------------------
# 1. TARGET VARIABLE ANALYSIS (Vehicle_Unit_Rate)
# -------------------------------------------------------------
print("--- 1. Target Variable (Vehicle_Unit_Rate) Summary ---")
unit_rate_stats = df["Vehicle_Unit_Rate"].describe()
print(unit_rate_stats.round(2))
skewness = df["Vehicle_Unit_Rate"].skew()
print(f"Skewness: {skewness:.2f} (Values > 1 indicate right-skewed distribution)\n")

# -------------------------------------------------------------
# 2. UNIT RATE BY CATEGORICAL DRIVERS
# -------------------------------------------------------------
print("--- 2. Average Unit Rate by Vehicle Category ---")
cat_summary = df.groupby("Vehicle_Category")["Vehicle_Unit_Rate"].agg(
    Mean_Rate="mean", Median_Rate="median", Min_Rate="min", Max_Rate="max", Trip_Count="count"
).round(2).sort_values(by="Mean_Rate", ascending=False)
print(cat_summary)

print("\n--- 3. Average Unit Rate by Unit of Measure (Vehicle_UOM) ---")
uom_summary = df.groupby("Vehicle_UOM")["Vehicle_Unit_Rate"].agg(
    Mean_Rate="mean", Median_Rate="median", Trip_Count="count"
).round(2)
print(uom_summary)

print("\n--- 4. Average Unit Rate by ODC Flag (Over Dimensional Cargo) ---")
odc_summary = df.groupby("ODC_Flag")["Vehicle_Unit_Rate"].agg(
    Mean_Rate="mean", Median_Rate="median", Trip_Count="count"
).round(2)
print(odc_summary)

print("\n--- 5. Average Unit Rate by Vendor ---")
vendor_summary = df.groupby("Vendor_Name")["Vehicle_Unit_Rate"].agg(
    Mean_Rate="mean", Trip_Count="count"
).round(2).sort_values(by="Mean_Rate", ascending=False)
print(vendor_summary)

# -------------------------------------------------------------
# 3. CORRELATION WITH OPERATIONAL FEATURES (No Data Leakage)
# -------------------------------------------------------------
print("\n--- 6. Correlation with Vehicle_Unit_Rate ---")
# Operational features only (Excluding Freight_Charge, Detention_Charge, Final_Total_Amount)
operational_numeric_cols = [
    "Distance_Km",
    "Vehicle_Capacity",
    "Vehicle_Length",
    "Vehicle_Width",
    "Vehicle_Height",
    "Vehicle_Transit_Days",
    "Fuel_Price",
    "Vehicle_Unit_Quantity",
    "Vehicle_Unit_Rate"  # Target
]

corr_matrix = df[operational_numeric_cols].corr()
target_corr = corr_matrix["Vehicle_Unit_Rate"].sort_values(ascending=False).round(4)
print(target_corr)

# -------------------------------------------------------------
# 4. GENERATE EDA VISUALIZATIONS
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Distribution of Unit Rate
sns.histplot(df["Vehicle_Unit_Rate"], kde=True, ax=axes[0, 0], color="royalblue")
axes[0, 0].set_title("Distribution of Vehicle Unit Rate", fontsize=12, fontweight="bold")
axes[0, 0].set_xlabel("Unit Rate (Rs.)")

# Subplot 2: Unit Rate across Vehicle Categories
sns.boxplot(x="Vehicle_Category", y="Vehicle_Unit_Rate", data=df, ax=axes[0, 1], palette="Set2")
axes[0, 1].set_title("Unit Rate by Vehicle Category", fontsize=12, fontweight="bold")
axes[0, 1].set_xlabel("Vehicle Category")
axes[0, 1].set_ylabel("Unit Rate (Rs.)")

# Subplot 3: Distance vs Unit Rate (colored by ODC Flag)
sns.scatterplot(
    x="Distance_Km", y="Vehicle_Unit_Rate", hue="ODC_Flag",
    data=df, ax=axes[1, 0], s=80, alpha=0.85
)
axes[1, 0].set_title("Distance (Km) vs Vehicle Unit Rate", fontsize=12, fontweight="bold")
axes[1, 0].set_xlabel("Distance (Km)")
axes[1, 0].set_ylabel("Unit Rate (Rs.)")

# Subplot 4: Operational Correlation Heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1, 1], square=True)
axes[1, 1].set_title("Operational Feature Correlation Heatmap", fontsize=12, fontweight="bold")

plt.tight_layout()
plot_filename = "eda_summary.png"
plt.savefig(plot_filename, dpi=300)
print(f"\nVisualization saved successfully as: '{plot_filename}'")

print("=== STAGE 2: EXPLORATORY DATA ANALYSIS COMPLETED ===")