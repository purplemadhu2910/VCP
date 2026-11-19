"""
STAGE 6: TRAIN-TEST SPLIT
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load the encoded dataset
INPUT_PATH = "encoded_tc_dataset.csv"
df = pd.read_csv(INPUT_PATH)

print("=== STAGE 6: TRAIN-TEST SPLIT START ===")

# 2. Separate Features (X) and Target (y)
TARGET_COL = "Vehicle_Unit_Rate"

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

print(f"Total Feature Matrix (X) shape: {X.shape}")
print(f"Total Target Vector (y) shape:  {y.shape}")

# 3. Perform 80-20 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# 4. Verify Shapes of Split Datasets
print("\n--- Split Dataset Dimensions ---")
print(f"Training Features (X_train): {X_train.shape} (80% of data)")
print(f"Testing Features  (X_test):  {X_test.shape}  (20% of data)")
print(f"Training Target   (y_train): {y_train.shape}")
print(f"Testing Target    (y_test):  {y_test.shape}")

# 5. Sanity Check: Compare Target Distribution (Train vs Test)
print("\n--- Target Variable Distribution Check (Train vs Test) ---")
distribution_check = pd.DataFrame({
    "Metric": ["Mean (Rs.)", "Median (Rs.)", "Std Dev (Rs.)", "Min (Rs.)", "Max (Rs.)"],
    "Train (y_train)": [
        round(y_train.mean(), 2),
        round(y_train.median(), 2),
        round(y_train.std(), 2),
        round(y_train.min(), 2),
        round(y_train.max(), 2)
    ],
    "Test (y_test)": [
        round(y_test.mean(), 2),
        round(y_test.median(), 2),
        round(y_test.std(), 2),
        round(y_test.min(), 2),
        round(y_test.max(), 2)
    ]
})
print(distribution_check.to_string(index=False))

# 6. Save Split Datasets for Model Training & Evaluation
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("\n✓ Split datasets successfully saved to CSV:")
print("  - X_train.csv")
print("  - X_test.csv")
print("  - y_train.csv")
print("  - y_test.csv")
print("=== STAGE 6: TRAIN-TEST SPLIT COMPLETED ===")