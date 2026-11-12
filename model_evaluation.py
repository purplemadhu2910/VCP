"""
STAGE 8: MODEL EVALUATION
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plotting style
sns.set_theme(style="whitegrid")

print("=== STAGE 8: MODEL EVALUATION (ON UNSEEN TEST DATA) START ===")

# 1. Load Unseen Test Data and Saved Models
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv").values.ravel()
trained_models = joblib.load("trained_models.pkl")

print(f"Evaluating across {len(y_test)} unseen test records...")

# 2. Evaluate Each Model on X_test
evaluation_results = []
predictions_dict = {}

for name, model in trained_models.items():
    # Predict on test data
    y_pred = model.predict(X_test)
    predictions_dict[name] = y_pred
    
    # Calculate performance metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    evaluation_results.append({
        "Model": name,
        "Test R²": round(r2, 4),
        "MAE (Rs.)": round(mae, 2),
        "RMSE (Rs.)": round(rmse, 2),
        "MAPE (%)": round(mape, 2)
    })

# 3. Display Formatted Evaluation Table
eval_df = pd.DataFrame(evaluation_results).sort_values(by="Test R²", ascending=False)
print("\n--- Test Set Performance Benchmark ---")
print(eval_df.to_string(index=False))

# 4. Save Evaluation Metrics to CSV for Model Comparison (Stage 9)
eval_df.to_csv("evaluation_results.csv", index=False)
print("\n✓ Evaluation results saved to 'evaluation_results.csv'")

# 5. Inspect Sample Actual vs Predicted Rates (First 5 Test Trips)
print("\n--- Sample Predictions Comparison (First 5 Test Trips) ---")
sample_comparison = pd.DataFrame({"Actual Rate (Rs.)": np.round(y_test[:5], 2)})
for name, preds in predictions_dict.items():
    sample_comparison[f"Pred: {name}"] = np.round(preds[:5], 2)
print(sample_comparison.to_string(index=False))

# 6. Generate Actual vs Predicted Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

rf_preds = predictions_dict["Random Forest"]
gb_preds = predictions_dict["Gradient Boosting"]

# Left Subplot: Random Forest Actual vs Predicted
axes[0].scatter(y_test, rf_preds, color="forestgreen", edgecolor="black", alpha=0.8, s=80)
min_val = min(y_test.min(), rf_preds.min())
max_val = max(y_test.max(), rf_preds.max())
axes[0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
axes[0].set_title(f"Random Forest (Test R²: {eval_df.loc[eval_df['Model']=='Random Forest', 'Test R²'].values[0]})", fontweight="bold")
axes[0].set_xlabel("Actual Unit Rate (Rs.)")
axes[0].set_ylabel("Predicted Unit Rate (Rs.)")
axes[0].legend()

# Right Subplot: Gradient Boosting Actual vs Predicted
axes[1].scatter(y_test, gb_preds, color="darkorange", edgecolor="black", alpha=0.8, s=80)
min_val = min(y_test.min(), gb_preds.min())
max_val = max(y_test.max(), gb_preds.max())
axes[1].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit (y = x)")
axes[1].set_title(f"Gradient Boosting (Test R²: {eval_df.loc[eval_df['Model']=='Gradient Boosting', 'Test R²'].values[0]})", fontweight="bold")
axes[1].set_xlabel("Actual Unit Rate (Rs.)")
axes[1].set_ylabel("Predicted Unit Rate (Rs.)")
axes[1].legend()

plt.tight_layout()
eval_plot = "model_evaluation_plot.png"
plt.savefig(eval_plot, dpi=300)
print(f"✓ Visual evaluation plot saved as: '{eval_plot}'")

print("=== STAGE 8: MODEL EVALUATION COMPLETED ===")