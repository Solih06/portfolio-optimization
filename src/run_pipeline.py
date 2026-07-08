import os
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from src.data_preprocessing import fetch_raw_data, clean_market_data, transform_to_log_returns, isolate_anomalous_outliers
from src.forecasting_engine import split_data_chronologically, optimize_and_forecast_arima, compute_forecast_accuracies

def execute_production_pipeline():
    print("🚀 Initializing GMF Quantitative Asset Management Pipeline...")
    
    # 1. Ingestion and Preprocessing
    raw_prices = fetch_raw_data(tickers=["TSLA", "SPY", "BND"], start_date="2015-01-01", end_date="2026-06-30")
    cleaned_prices = clean_market_data(raw_prices)
    log_returns = transform_to_log_returns(cleaned_prices)
    
    # 2. Run Stationarity Validation Checks via ADF
    print("\n📊 Running Augmented Dickey-Fuller (ADF) Test Suite...")
    for ticker in log_returns.columns:
        adf_result = adfuller(log_returns[ticker])
        print(f"Asset [{ticker}] Log Returns ADF Stat: {adf_result[0]:.4f} | p-value: {adf_result[1]:.4f}")
    
    # 3. Structural Outlier Auditing
    outliers = isolate_anomalous_outliers(log_returns)
    for ticker, series in outliers.items():
        print(f"⚠️ isolated {len(series)} statistical tail outliers for asset ticker: {ticker}")
        
    # 4. Chronological Splitting & Forecasting Target Selection (e.g., SPY Benchmark)
    target_asset = "SPY"
    train_set, test_set = split_data_chronologically(log_returns[target_asset], split_date="2025-01-01")
    
    # 5. ARIMA Time Series Execution
    print(f"\n🔮 Training ARIMA(1,0,1) baseline configuration on {target_asset} training returns...")
    predictions, model_fit = optimize_and_forecast_arima(train_set, test_set, order=(1, 0, 1))
    
    # 6. Performance Evaluation Metric Extraction
    metrics = compute_forecast_accuracies(test_set, predictions)
    print(f"\n📈 Out-of-Sample Validation Diagnostics [{target_asset}]:")
    for key, val in metrics.items():
        print(f" - {key}: {val:.6f}")
        
    print("\n✅ Pipeline execution run finalized with zero fatal exit metrics.")

if __name__ == "__main__":
    execute_production_pipeline()