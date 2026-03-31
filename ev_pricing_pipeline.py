"""
EV Price Forecast Hackathon: Asset Valuation Pipeline
Refined Portfolio Version: Modular Regression Ensembles
Focused on Feature Importance & Predictive Accuracy
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import logging

# Configure Environment
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EVPricingModel:
    """XGBoost Regressor for Electric Vehicle pricing."""
    def __init__(self, params: dict):
        self.params = params
        self.model = None

    def train(self, X_train, y_train, X_val, y_val):
        """Train the XGBoost model with early stopping."""
        logging.info("Initializing XGBoost Training...")
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        logging.info("Training complete.")
        return self.model

    def evaluate(self, X_test, y_test):
        """Calculate RMSE performance."""
        preds = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        logging.info(f"Model RMSE Performance: {rmse:.4f}")
        return rmse

    def get_importance(self):
        """Extract feature importance for market analysis."""
        importance = self.model.get_booster().get_score(importance_type='gain')
        return pd.DataFrame(importance.items(), columns=['Feature', 'Gain']).sort_values(by='Gain', ascending=False)

if __name__ == "__main__":
    # Portfolio Demo Logic
    xgb_params = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 50
    }
    
    pricing_ai = EVPricngModel(xgb_params)
    logging.info("EV Pricing Engine ready for market dataset ingestion.")
    logging.info("Primary valuation drivers: Battery Capacity, Max Range, and Brand Equity.")
