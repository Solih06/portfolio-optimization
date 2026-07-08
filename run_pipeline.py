import os
import pandas as pd
from src.data_preprocessing import fetch_raw_data, clean_market_data, transform_to_log_returns
from src.forecasting_engine import split_data_chronologically, optimize_and_forecast_arima, compute_forecast_accuracies
from src.deep_learning_models import train_and_forecast_lstm
from src.portfolio_optimizer import optimize_max_sharpe, calculate_portfolio_performance

def execute_production_pipeline():
    print("🚀 Initializing GMF Quantitative Multi-Model Forecasting & Allocation Pipeline...")
    
    # 1. Pipeline Ingestion & Preprocessing
    tickers = ["TSLA", "SPY", "BND"]
    raw_prices = fetch_raw_data(tickers=tickers, start_date="2015-01-01", end_date="2026-06-30")
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
    print(f"\n🧠 Training Stacked LSTM Architecture on {target_asset}...")
    lstm_preds = train_and_forecast_lstm(train_set, test_set, sequence_length=30, epochs=5, batch_size=64)
    lstm_metrics = compute_forecast_accuracies(test_set, lstm_preds)
    
    # 5. Model Execution: Portfolio Allocation Optimization
    print(f"\n⚖️ Executing Modern Portfolio Theory Allocation Optimizer across {tickers}...")
    optimal_weights = optimize_max_sharpe(log_returns, risk_free_rate=0.02)
    
    # Extract performance metrics for the optimal allocation
    p_ret, p_vol, p_sharpe = calculate_portfolio_performance(
        optimal_weights, log_returns.mean(), log_returns.cov(), risk_free_rate=0.02
    )
    
    # 6. Comprehensive Reporting Engine Logs
    print("\n" + "="*60)
    print(f"📈 SECTION 1: MODEL COMPARISON KEY PERFORMANCE LOGS [{target_asset}]")
    print("="*60)
    print(f"Metric   | ARIMA Baseline | Deep Learning LSTM")
    print("-"*60)
    print(f"MAE      | {arima_metrics['MAE']:.6f}       | {lstm_metrics['MAE']:.6f}")
    print(f"RMSE     | {arima_metrics['RMSE']:.6f}       | {lstm_metrics['RMSE']:.6f}")
    print(f"MAPE (%) | {arima_metrics['MAPE']:.4f}%      | {lstm_metrics['MAPE']:.4f}%")
    
    print("\n" + "="*60)
    print("💼 SECTION 2: MAXIMUM SHARPE RATIO OPTIMAL ALLOCATION")
    print("="*60)
    for asset, weight in zip(tickers, optimal_weights):
        print(f"Asset: {asset:<6} | Targeted Allocation Weight: {weight*100:6.2f}%")
    print("-"*60)
    print(f"Expected Annualized Portfolio Return:     {p_ret*100:.2f}%")
    print(f"Expected Annualized Portfolio Volatility: {p_vol*100:.2f}%")
    print(f"Optimized Portfolio Sharpe Ratio:         {p_sharpe:.4f}")
    print("="*60)
    print("\n✅ Multi-model allocation pipeline run successfully finalized.")

if __name__ == "__main__":
    execute_production_pipeline()