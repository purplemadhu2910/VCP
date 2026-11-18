"""
STAGE 9: MODEL COMPARISON & CHAMPION SELECTION
Project: Vehicle Transportation Unit Rate Prediction (L&T)
Target (Y): Vehicle_Unit_Rate
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure working directory is the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

sns.set_theme(style="whitegrid")

print("=== STAGE 9: MODEL COMPARISON START ===")

# 1. Load Evaluation Results & Trained Models
eval_df = pd.read_csv("evaluation_results.csv")
trained_models = joblib.load("trained_models.pkl")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()

# 2. Add Training Scores & Generalization Gap
train_scores = []
for _, row in eval_df.iterrows():
    model_name = row["Model"]
    train_r2 = trained_models[model_name].score(X_train, y_train)
    train_scores.append(round(train_r2, 4))

eval_df["Train R²"] = train_scores
eval_df["Generalization Gap"] = (eval_df["Train R²"] - eval_df["Test R²"]).round(4)

# Reorder columns for presentation
comparison_table = eval_df[["Model", "Train R²", "Test R²", "Generalization Gap", "MAE (Rs.)", "RMSE (Rs.)", "MAPE (%)"]]
print("\n--- Comprehensive Model Comparison ---")
print(comparison_table.to_string(index=False))

# 3. Identify and Crown Champion Model
best_row = eval_df.sort_values(by="Test R²", ascending=False).iloc[0]
champion_name = best_row["Model"]
champion_model = trained_models[champion_name]

print(f"\n🏆 CHAMPION MODEL SELECTED: {champion_name}")
print(f"   - Test R²:   {best_row['Test R²']}")
print(f"   - Test MAE:  Rs. {best_row['MAE (Rs.)']}")
print(f"   - Test MAPE: {best_row['MAPE (%)']}%")

# Save the Champion Model separately for inference
joblib.dump(champion_model, "champion_model.pkl")
print("✓ Champion model saved as: 'champion_model.pkl'")

# 4. Feature Importance Analysis (What did the Champion Model learn?)
print(f"\n--- Top 10 Most Important Cost Drivers ({champion_name}) ---")
if hasattr(champion_model, "feature_importances_"):
    importances = pd.Series(champion_model.feature_importances_, index=X_train.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    
    for rank, (feat, score) in enumerate(top_features.items(), 1):
        print(f"{rank:2d}. {feat:<30} Importance: {score:.4f} ({score*100:.1f}%)")
else:
    top_features = None

# 5. Extract and Export Regression Coefficients (Ridge Model)
print("\n--- Regression Feature Coefficients (Ridge Model) ---")
if "Ridge Regression" in trained_models:
    ridge_step = trained_models["Ridge Regression"].named_steps["regressor"]
    coef_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": ridge_step.coef_.round(2)
    }).sort_values(by="Coefficient", ascending=False)
    
    print(coef_df.to_string(index=False))
    coef_df.to_csv("model_coefficients.csv", index=False)
    print("✓ All feature coefficients saved to: 'model_coefficients.csv'")

# 6. Visual Model Comparison Plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left Plot: Test R² vs MAE Benchmark
ax1 = axes[0]
palette = ["#2ecc71" if m == champion_name else "#3498db" for m in comparison_table["Model"]]
bars = ax1.bar(comparison_table["Model"], comparison_table["Test R²"], color=palette, edgecolor="black", width=0.55)
ax1.set_title("Model Accuracy Benchmark (Test R²)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Test R² Score")
ax1.set_ylim(0, 1.05)

for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

# Right Plot: Top 8 Feature Importances of Champion Model
ax2 = axes[1]
if top_features is not None:
    top_8 = top_features.head(8).sort_values(ascending=True)
    ax2.barh(top_8.index, top_8.values * 100, color="#e67e22", edgecolor="black")
    ax2.set_title(f"Top 8 Cost Drivers ({champion_name})", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Relative Importance (%)")

plt.tight_layout()
comparison_plot = "model_comparison_summary.png"
plt.savefig(comparison_plot, dpi=300)
print(f"\n✓ Comparison visualizations saved as: '{comparison_plot}'")

print("=== STAGE 9: MODEL COMPARISON COMPLETED ===")