# Commercial Vehicle Transportation Unit Rate Prediction & Corridor Mapping Portal

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Gradient%20Boosting-green.svg)](https://scikit-learn.org/)
[![UI](https://img.shields.io/badge/Frontend-Streamlit%20%7C%20Folium-red.svg)](https://streamlit.io/)
[![Dataset](https://img.shields.io/badge/Dataset-10%2C000%20Trips-orange.svg)]()
[![Model Accuracy](https://img.shields.io/badge/Test%20R%C2%B2-98.57%25-brightgreen.svg)]()
[![MAE](https://img.shields.io/badge/MAE-%E2%82%B9502.19-teal.svg)]()

An end-to-end Machine Learning regression engine and interactive geographic corridor portal developed to predict the **fair baseline transportation unit rate** (`Vehicle_Unit_Rate`) for commercial freight operations across India.

Trained and evaluated on **10,000 real-world freight trips**, the system automates commercial freight rate benchmarking, detects vendor overcharging, factors in Over-Dimensional Cargo (ODC) regulatory surcharges, and visualizes inter-state freight transit corridors on an interactive GIS map.
Live Link: **https://cvturp.streamlit.app/**

---

## Executive Highlights

- **Champion Algorithm:** Gradient Boosting Regressor (100 sequential decision trees, learning rate 0.08, max depth 3).
- **Test Accuracy ($R^2$):** **98.57%** evaluated on 2,000 completely unseen holdout trips.
- **Mean Absolute Error (MAE):** **₹502.19** on commercial trips averaging over ₹7,000 per unit.
- **Mean Absolute Percentage Error (MAPE):** **12.06%**.
- **Generalization Gap:** Only **0.52% (0.0052)** between training ($99.09\%$) and testing ($98.57\%$), confirming zero overfitting.
- **Strict Data Leakage Prevention:** All post-transit billing fields (`Vehicle_Freight_Charge`, `Vehicle_Detention_Charge`, `Vehicle_Final_Total_Amount`, `Vehicle_Detention_Days`) were strictly excluded from model inputs to ensure 100% pre-dispatch validity.
- **Interactive Route Portal:** Two-column Streamlit dashboard integrating dynamic Leaflet/OpenStreetMap mapping across 21 major Indian logistics hubs with zero external API key dependencies.

---

## Key System Features

### 1. Interactive Geographic Freight Corridor Map
- **21 Indian Logistics Hubs:** Pre-configured GPS coordinate database spanning Ahmedabad, Mumbai, Delhi, Bengaluru, Chennai, Kolkata, Hyderabad, Pune, Surat, Vadodara, Jaipur, Lucknow, Nagpur, and more.
- **Dynamic Visual Route:** Automatically drops an Origin marker (pickup hub) and a Destination marker (delivery warehouse) connected by a dashed royal-blue transit polyline.
- **Auto-Fit Camera Bounds:** Automatically zooms and pans to frame the entire highway route whether the corridor is 110 Km (Vadodara ➔ Ahmedabad) or 2,100 Km (Delhi ➔ Chennai).
- **Zero API Dependency:** Powered by Python `folium` and OpenStreetMap tiles; runs completely offline/free without paid Google Cloud billing.

### 2. Operational & Commercial Trip Input Form
- **Route Details:** Origin & Destination searchable dropdowns with same-city validation alerts, route distance auto-suggestions, transit day calculations, and inquiry date indexation.
- **Vehicle & Cargo Specs:** Classifications across Multi-Axle Trucks, Taurus (16–25T), SLBT/LBT Low Bed Trailers, and LCVs; 3D trailer dimensions (Length, Height, Width) with real-time cubic feet calculation; payload capacity in Metric Tons.
- **Regulatory ODC Surcharges:** Toggle for Over Dimensional Cargo (ODC) requiring special state permits, escort coordination, and reduced travel speed.
- **Commercial Indexation:** Consignment volume brackets, 3PL carrier selection, and real-time state diesel benchmark price tracking.

### 3. Official Freight Quotation Generation
- **Quotation ID:** Generates an enterprise reference ID (`QT-2026-XXXX`).
- **Financial Metrics:** Fair Unit Rate (₹/MT or ₹/Nos), Estimated Total Freight (₹), and Daily Travel Velocity (Km/day).
- **Commercial Summary Table:** Complete operational bill of attributes.
- **Rule-Based Pricing Intelligence:** Automated badges explaining bulk economies of scale, ODC permit surcharges, and express multi-driver transit requirements.
- **Audit-Ready Export:** One-click CSV quotation download for contract negotiations and procurement records.

---

## 10-Stage Machine Learning Pipeline

```
Raw Data (tc_dataset.csv - 10,000 records)
   │
   ├── 01. Data Cleaning (data_cleaning.py)
   │      └── Handles null values, validates physical ranges (Distance > 0, Dimensions > 0)
   │      └── Output: cleaned_tc_dataset.csv
   │
   ├── 02. Exploratory Data Analysis (eda.py)
   │      └── Target distributions, category boxplots, correlation heatmaps
   │      └── Output: eda_summary.png
   │
   ├── 03. Feature Engineering (feature_engineering.py)
   │      └── Derives: Vehicle_Volume, Capacity_per_Volume, Km_Per_Day, Created_Month
   │      └── Output: engineered_tc_dataset.csv
   │
   ├── 04. Feature Selection (feature_selection.py)
   │      └── Eliminates post-transit leakages & high-cardinality lane identifiers
   │      └── Retains 12 operational pre-dispatch predictors
   │      └── Output: selected_features_tc_dataset.csv
   │
   ├── 05. Categorical Encoding (categorical_encoding.py)
   │      └── One-Hot Encoding (drop_first=True) to prevent dummy variable trap
   │      └── Output: encoded_tc_dataset.csv (23 numerical columns)
   │
   ├── 06. Train-Test Split (train_test_split.py)
   │      └── 80-20 stratified holdout split (random_state=42)
   │      └── Output: X_train.csv (8,000), X_test.csv (2,000), y_train.csv, y_test.csv
   │
   ├── 07. Model Training (model_training.py)
   │      └── Fits Linear Regression, Ridge, Random Forest, and Gradient Boosting
   │      └── Output: trained_models.pkl
   │
   ├── 08. Model Evaluation (model_evaluation.py)
   │      └── Computes R², MAE, RMSE, MAPE on 2,000 unseen holdout trips
   │      └── Output: evaluation_results.csv, model_evaluation_plot.png
   │
   ├── 09. Model Comparison & Champion Selection (model_comparision.py)
   │      └── Evaluates generalization gaps; exports champion model and Ridge coefficients
   │      └── Output: champion_model.pkl, model_coefficients.csv, model_comparison_summary.png
   │
   └── 10. Live Inference & Web Dashboard
          ├── main_pipeline.py       -> Master 1-click execution of stages 1–9
          ├── predict_new_data.py    -> Batch inference script
          ├── test_on_new_dataset.py -> Automated validation on fresh trip inquiries
          └── app.py                 -> Interactive Streamlit Web UI with GIS Corridor Map
```

---

## Model Benchmark Results (Evaluated on 2,000 Unseen Trips)

| Algorithm | Training $R^2$ | Test $R^2$ | Generalization Gap | MAE (₹) | RMSE (₹) | MAPE (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **0.9909** | **0.9857** | **0.0052 (0.5%)** | **₹502.19** | **₹903.88** | **12.06%** | **Champion** |
| **Random Forest** | 0.9758 | 0.9636 | 0.0122 (1.2%) | ₹850.31 | ₹1,444.36 | 15.38% | Runner-Up |
| **Linear Regression** | 0.5499 | 0.6120 | -0.0621 | ₹2,746.49 | ₹4,715.06 | 94.89% | Failed (Negative Rates) |
| **Ridge Regression** | 0.5498 | 0.6114 | -0.0616 | ₹2,747.72 | ₹4,718.59 | 94.78% | Underfitted |

### Why Gradient Boosting Won:
Transportation pricing is inherently non-linear: economies of scale taper off, dimensional volume interacts with vehicle categories, and ODC cargo adds discontinuous permit step-costs. Linear and regularized models failed completely (MAPE > 94%), while Gradient Boosting captured non-linear interactions sequentially with an average absolute error of just ₹502.19.

---

## Supply Chain & Commercial Insights

1. **Economies of Scale Dominate (62.4% Feature Importance):**
   - `Vehicle_Unit_Quantity` is the single largest determinant of per-unit pricing. High-tonnage shipments receive steep volume discounts, reducing unit transport costs by up to 55%.
2. **Transit Distance Sets Operating Baseline (33.6% Feature Importance):**
   - Line-haul distance dictates fuel consumption, driver allowances, toll charges, and vehicle turnaround schedules.
   - *Together, Consignment Quantity and Distance account for **96%** of all pricing variance.*
3. **Over-Dimensional Cargo (ODC) Premium:**
   - Cargo exceeding standard trailer widths or heights requires state transit clearances, escort vehicles, and daytime-only travel restrictions, commanding an average **61% pricing surcharge** over standard containerized trucks.

---

## Repository Structure

```
VCP/
├── tc_dataset.csv                # Raw historical logistics dataset (10,000 trips)
├── data_cleaning.py              # Stage 1: Data cleaning & schema validation
├── eda.py                        # Stage 2: Exploratory data analysis & plots
├── feature_engineering.py        # Stage 3: Engineering volume, velocity, and date features
├── feature_selection.py          # Stage 4: Leakage prevention and feature filtering
├── categorical_encoding.py       # Stage 5: One-hot encoding (23 features)
├── train_test_split.py           # Stage 6: 80/20 train-test holdout split
├── model_training.py             # Stage 7: Fitting 4 candidate regression models
├── model_evaluation.py           # Stage 8: Holdout benchmarking (R², MAE, RMSE, MAPE)
├── model_comparision.py          # Stage 9: Generalization analysis & champion serialization
├── main_pipeline.py              # Master pipeline runner (executes stages 1–9 sequentially)
├── test_on_new_dataset.py        # Automated testing script on new data batches
├── predict_new_data.py           # Batch inference script for new inquiries
├── app.py                        # Interactive Streamlit frontend with GIS Corridor Map
├── champion_model.pkl            # Serialized Gradient Boosting champion model
├── model_coefficients.csv        # Linear/Ridge standardized feature coefficients
├── predictions_output.csv        # Batch prediction outputs
├── X_train.csv, y_train.csv      # Processed training features and target labels
├── X_test.csv, y_test.csv        # Processed testing features and target labels
├── eda_summary.png               # EDA visual dashboard
├── model_evaluation_plot.png     # Actual vs Predicted evaluation plot
├── model_comparison_summary.png  # Model accuracy and feature importance summary
├── requirements.txt              # Production Python dependencies
└── README.md                     # Project documentation
```

---

## Quick Start Guide

### 1. Set Up Environment
```powershell
# Navigate to project folder
cd VCP

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full ML Pipeline (Stages 1 to 9)
To retrain, evaluate, and benchmark all models from raw data with a single command:
```powershell
python main_pipeline.py
```

### 3. Launch the Interactive Web Dashboard
```powershell
python -m streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to access the rate engine and interactive route map.

### 4. Run Batch Inferences on New Trip Files
```powershell
python predict_new_data.py
```

---

## Sample Web Portal Inferences

| Route Scenario | Vehicle Type | Distance | Load Quantity | ODC | Predicted Unit Rate | Estimated Total Freight |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Industrial Long-Haul** (Ahmedabad ➔ Mumbai) | Multi-Axle Truck (32x8x8 ft) | 530 Km | 18.0 MT | Non-ODC | **₹4,284.10 / MT** | **₹77,113.80** |
| **Heavy Machinery Transit** (Surat ➔ Delhi) | SLBT Low Bed Trailer | 1,180 Km | 2.0 Nos | ODC | **₹13,850.50 / Unit** | **₹27,701.00** |
| **Local Regional Feeder** (Vadodara ➔ Surat) | LCV Light Commercial | 155 Km | 5.0 MT | Non-ODC | **₹1,640.25 / MT** | **₹8,201.25** |

---

## Technical Specifications
- **Programming Language:** Python 3.11+
- **Machine Learning Libraries:** Scikit-Learn, NumPy, Pandas, Joblib
- **Frontend & GIS Visualization:** Streamlit, Folium, Streamlit-Folium, OpenStreetMap
- **Evaluation Metrics:** Coefficient of Determination ($R^2$), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE)
