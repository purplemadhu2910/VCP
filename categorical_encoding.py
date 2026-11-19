"""
STAGE 5: ENCODING CATEGORICAL VARIABLES
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np

# 1. Load the selected features dataset
INPUT_PATH = "selected_features_tc_dataset.csv"
OUTPUT_PATH = "encoded_tc_dataset.csv"

df = pd.read_csv(INPUT_PATH)
print("=== STAGE 5: ENCODING CATEGORICAL VARIABLES START ===")
print(f"Dataset shape before encoding: {df.shape}")

# 2. Identify Categorical Columns
categorical_cols = ["Vehicle_Category", "ODC_Flag", "Vehicle_UOM", "Vendor_Name"]
target_col = "Vehicle_Unit_Rate"

print("\nCategories per feature before encoding:")
for col in categorical_cols:
    unique_vals = df[col].unique().tolist()
    print(f"- {col} ({len(unique_vals)} unique): {unique_vals}")

# 3. Apply One-Hot Encoding with drop_first=True
# dtype=int ensures columns are created as clean 0 and 1 integers
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

# 4. Verify the Encoded Features
print(f"\nDataset shape after encoding: {df_encoded.shape}")
print(f"Total features created: {df_encoded.shape[1] - 1} (Excluding target)")

print("\n--- All Column Names in the Encoded Dataset ---")
for i, col in enumerate(df_encoded.columns, 1):
    tag = " <-- TARGET (Y)" if col == target_col else ""
    print(f"{i:2d}. {col}{tag}")

# 5. Check Data Types (Ensure 100% numerical values)
non_numeric = df_encoded.select_dtypes(exclude=[np.number]).columns
if len(non_numeric) == 0:
    print("\n✓ Verification Passed: All columns are now fully numeric (float/int).")
else:
    print(f"\n⚠️ Non-numeric columns remaining: {list(non_numeric)}")

# 6. Save Encoded Dataset
df_encoded.to_csv(OUTPUT_PATH, index=False)
print(f"\nEncoded dataset saved successfully as: '{OUTPUT_PATH}'")
print("=== STAGE 5: ENCODING COMPLETED ===")