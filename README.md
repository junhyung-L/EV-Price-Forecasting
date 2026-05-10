# ⚡ EV Price Forecast Hackathon: Data-Driven Asset Valuation

## 📌 Project Overview
As the Electric Vehicle (EV) market matures, accurate asset valuation becomes critical for consumers and manufacturers alike. This repository contains the solution for the **EV Price Forecast Hackathon** hosted by DACON. 

The goal is to predict EV prices based on technical specifications and market data. This project achieved an **RMSE of 0.919** (Normalized Scale) on the leaderboard using a high-precision **XGBoost-based regression model**.

---

## 📊 1. Data & Preprocessing (데이터 및 전처리)

The project handles a dataset containing technical specifications of various EV models. Key highlights include advanced feature engineering and machine learning-based imputation.

### 🧠 Advanced Feature Engineering (고급 피처 엔지니어링)
To capture the complex relationships in vehicle pricing, several domain-specific features were engineered:
- **Vehicle Efficiency (`전비(km/kWh)`)**: Calculated as `주행거리(km) / 배터리용량` to represent energy efficiency.
- **Mileage Ratio (`주행거리비율`)**: Annual mileage calculated by dividing total mileage by age.
- **Combined Category (`제조사_모델_상태`)**: Interaction feature combining Manufacturer, Model, and Vehicle Condition to capture brand equity and depreciation simultaneously.
- **Usage Weight**: Mapping weights for drive types (AWD, RWD, FWD) and accident history.

### 🛠️ Predictive Imputation for Missing Battery Capacity
- **Problem**: A significant portion of the `배터리용량` (Battery Capacity) data was missing.
- **Solution**: Instead of simple mean imputation, an **XGBRegressor** was trained on non-missing data to predict the missing battery capacities based on other features like manufacturer, model, and mileage.
- **Impact**: Preserved the integrity of the most critical feature in EV pricing.

### 🔬 Statistical Validation
- **ANOVA**: Used to verify the statistical significance of categorical features (Manufacturer, Model, Condition) against the target price.
- **Pearson Correlation**: Used to analyze the linear relationship between continuous variables and price, confirming strong negative correlation with mileage and positive correlation with battery capacity.

---

## 🤖 2. Modeling & Evaluation (모델링 및 평가)

- **Algorithm**: XGBoost Regressor.
- **Strategy**: Early stopping was used to prevent overfitting, and hyperparameters were tuned to balance bias and variance.
- **Evaluation**: The model successfully captured the nonlinear depreciation curves of EVs.

---

## 📁 Repository Structure
```text
├── data/                       # Dataset files
├── notebooks/                  # Exploratory notebooks
│   └── EV_price_prediction_xgb.ipynb
├── src/                        # Extracted Python scripts
│   └── ev_price_prediction_xgb.py
├── ev_pricing_pipeline.py      # Main pipeline script
└── README.md                   # Project documentation
```

---

## 🏁 3. Future Work
- Explore deep learning models for tabular data like **TabNet**.
- Implement **SHAP** to explain the non-linear impact of battery degradation on price.

---

## 👥 Contributors
- **Junhyung L.** (Project Lead / Data Scientist)

---
*Refactored and polished to meet professional software engineering standards for the [Data Analyst Portfolio](https://github.com/junhyung-L).*
*Note: Statistical findings and feature importances are based on the actual competition report results.*
