import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def split_data_chronologically(df, split_date="2025-01-01"):
    """
    Partitions the historical timeline cleanly into Train and Test blocks to prevent lookahead bias.
    """
    train = df[df.index < split_date]
    test = df[df.index >= split_date]
    return train, test

def optimize_and_forecast_arima(train_series, test_series, order=(1, 1, 1)):
    """
    Fits an ARIMA model on training logs and outputs predictions across the out-of-sample window.
    """
    # Note: We fit on log returns directly, so d=0 or 1 depending on further differencing needs.
    # To model raw price trend via log return forecasting, we forecast log returns and reconstruct.
    model = ARIMA(train_series, order=order)
    fitted_model = model.fit()
    
    # Generate predictions across the length of the test sequence
    forecasts = fitted_model.forecast(steps=len(test_series))
    forecasts.index = test_series.index
    return forecasts, fitted_model

def compute_forecast_accuracies(actual, predicted):
    """
    Calculates operational out-of-sample tracking metrics: MAE, RMSE, and MAPE.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Handle potential zero division safety bounds for MAPE
    pct_errors = np.abs((actual - predicted) / np.where(actual == 0, 1e-5, actual))
    mape = np.mean(pct_errors) * 100
    
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}