import os
import pandas as pd
from src.data_preprocessing import fetch_raw_data, clean_market_data, transform_to_log_returns
from src.forecasting_engine import split_data_chronologically, optimize_and_forecast_arima, compute_forecast_accuracies
from src.deep_learning_models import train_and_forecast_lstm

def execute_production_pipeline():
    print("🚀 Initializing GMF Quantitative Multi-Model Forecasting Pipeline...")
    
    # 1. Pipeline Ingestion & Preprocessing
    raw_prices = fetch_raw_data(tickers=["TSLA", "SPY", "BND"], start_date="2015-01-01", end_date="2026-06-30")
    cleaned_prices = clean_market_data(raw_prices)
    log_returns = transform_to_log_returns(cleaned_prices)
    
    # 2. Chronological Splitting (Targeting TSLA as our volatile asset)
    target_asset = "TSLA"
    train_set, test_set = split_data_chronologically(log_returns[target_asset], split_date="2025-01-01")
    
    # 3. Model Execution: Baseline ARIMA
    print(f"\n🔮 Fitting Baseline ARIMA(1,0,1) configuration on {target_asset}...")
    arima_preds, _ = optimize_and_forecast_arima(train_set, test_set, order=(1, 0, 1))
    arima_metrics = compute_forecast_accuracies(test_set, arima_preds)
    
    # 4. Model Execution: Deep Learning LSTM
    print(f"\n🧠 Training Stacked LSTM Architecture on {target_asset} (This may take a moment)...")
    lstm_preds = train_and_forecast_lstm(train_set, test_set, sequence_length=30, epochs=5, batch_size=64)
    lstm_metrics = compute_forecast_accuracies(test_set, lstm_preds)
    
    # 5. Comparative Evaluation Summary Table
    print("\n" + "="*50)
    print(f"📈 MODEL COMPARISON KEY PERFORMANCE LOGS [{target_asset}]")
    print("="*50)
    print(f"Metric   | ARIMA Baseline | Deep Learning LSTM")
    print("-"*50)
    print(f"MAE      | {arima_metrics['MAE']:.6f}       | {lstm_metrics['MAE']:.6f}")
    print(f"RMSE     | {arima_metrics['RMSE']:.6f}       | {lstm_metrics['RMSE']:.6f}")
    print(f"MAPE (%) | {arima_metrics['MAPE']:.4f}%      | {lstm_metrics['MAPE']:.4f}%")
    print("="*50)
    print("\n✅ Multi-model pipeline run successfully finalized.")

if __name__ == "__main__":
    execute_production_pipeline()