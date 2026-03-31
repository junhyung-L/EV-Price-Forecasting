# ⚡ EV Price Forecast Hackathon
## Data-Driven Asset Valuation for the Electric Vehicle Market

### 📌 Project Vision
As the EV market matures, accurate asset valuation becomes critical for consumers and manufacturers alike. This project develops a high-precision **XGBoost-based regression model** to predict EV prices, focusing on the nonlinear relationship between battery health, mileage, and brand equity.

### 🏆 Competition Highlights
- **Host:** DACON
- **Goal:** Predict EV prices based on technical specifications and market data.
- **Top Performance:** Achieved an **RMSE of 0.919** (Normalized Scale) on the leaderboard.

### 🛠️ Core Analytical Innovation
1. **Feature Engineering for Asset Depreciation:**
   - Isolated `Battery Capacity` and `Max Range` as the primary drivers of EV residual value.
   - Analyzed the inverse correlation between `Mileage` and `Price` to model depreciation curves.
2. **Gradient Boosted Regression (XGBoost):**
   - Implemented an **XGBoost Regressor** with Early Stopping to prevent overfitting.
   - Tuned hyperparameters (Learning Rate, Max Depth, Subsample) to balance model bias and variance.
3. **Advanced Feature Importance Analysis:**
   - Utilized XGBoost's `gain` and `weight` metrics to rank the technical features that most influence market pricing.

### 📊 Analytical ROI
- **Market Insight:** Discovered that battery capacity accounts for over 40% of the price variance in the current EV secondary market.
- **Scalability:** The pipeline is designed to be easily extended with **CatBoost** or **TabNet** for even higher predictive accuracy.

---
*This repository has been refined and modularized for the professional [Data Analyst Portfolio](https://github.com/junhyung-L).*
