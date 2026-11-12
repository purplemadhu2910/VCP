"""
STAGE 7: MODEL TRAINING
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

print("=== STAGE 7: MODEL TRAINING START ===")

# 1. Load Training Data
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()  # Flatten to 1D array

print(f"Loaded X_train shape: {X_train.shape}")
print(f"Loaded y_train shape: {y_train.shape}")

# 2. Define the 4 Candidate Models
# Note: For Linear and Ridge models, we use a Pipeline with StandardScaler
# to ensure features on large scales (Distance ~1500) don't dominate binary flags (0/1).
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
        n_estimators=100,
        max_depth=6,
        random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )
}

# 3. Fit (Train) Each Model on the Training Data
trained_models = {}
training_summary = []

print("\n--- Training Models on X_train (80 samples) ---")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Calculate training fit (R2 score on training set)
    train_r2 = model.score(X_train, y_train)
    
    trained_models[name] = model
    training_summary.append({
        "Model": name,
        "Training R² Score": round(train_r2, 4)
    })

# 4. Display Training Results
print("\n--- Training Fit Summary ---")
summary_df = pd.DataFrame(training_summary)
print(summary_df.to_string(index=False))

# 5. Save the Trained Models to Disk
# Using joblib so Stage 8 (Evaluation) and Stage 10 (Inference) can load them instantly
joblib.dump(trained_models, "trained_models.pkl")
print("\n✓ All 4 trained models saved successfully as: 'trained_models.pkl'")
print("=== STAGE 7: MODEL TRAINING COMPLETED ===")