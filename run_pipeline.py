import os
import pandas as pd
import numpy as np
from src.data_preprocessing import fetch_raw_data, clean_market_data, transform_to_log_returns
from src.forecasting_engine import split_data_chronologically, optimize_and_forecast_arima, compute_forecast_accuracies
from src.deep_learning_models import train_and_forecast_lstm
from src.portfolio_optimizer import optimize_max_sharpe, calculate_portfolio_performance
from src.backtest_engine import run_historical_backtest, compute_risk_metrics

def execute_production_pipeline():
    print("🚀 Initializing GMF Quantitative Multi-Model Forecasting, Allocation & Backtesting Pipeline...")
    
    # 1. Pipeline Ingestion & Preprocessing
    tickers = ["TSLA", "SPY", "BND"]
    raw_prices = fetch_raw_data(tickers=tickers, start_date="2015-01-01", end_date="2026-06-30")
    cleaned_prices = clean_market_data(raw_prices)
    log_returns = transform_to_log_returns(cleaned_prices)
    
    # 2. Chronological Splitting (Targeting TSLA)
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
    p_ret, p_vol, p_sharpe = calculate_portfolio_performance(
        optimal_weights, log_returns.mean(), log_returns.cov(), risk_free_rate=0.02
    )
    
    # 6. Model Execution: Backtesting Engine & Risk Metrics
    print("\n⏳ Simulating Historical Backtest and calculating risk metrics...")
    strat_equity, bench_equity, strat_ret, bench_ret = run_historical_backtest(log_returns, optimal_weights)
    strat_risk = compute_risk_metrics(strat_equity, strat_ret)
    bench_risk = compute_risk_metrics(bench_equity, bench_ret)
    
    # 7. Final Engineering Reporting Log
    print("\n" + "="*65)
    print(f"📈 SECTION 1: TARGET FORECAST PERFORMANCE LOGS [{target_asset}]")
    print("="*65)
    print(f"Metric   | ARIMA Baseline | Deep Learning LSTM")
    print("-"*65)
    print(f"MAE      | {arima_metrics['MAE']:.6f}       | {lstm_metrics['MAE']:.6f}")
    print(f"RMSE     | {arima_metrics['RMSE']:.6f}       | {lstm_metrics['RMSE']:.6f}")
    print(f"MAPE (%) | {arima_metrics['MAPE']:.4f}%      | {lstm_metrics['MAPE']:.4f}%")
    
    print("\n" + "="*65)
    print("💼 SECTION 2: MAXIMUM SHARPE RATIO TARGET ALLOCATION")
    print("="*65)
    for asset, weight in zip(tickers, optimal_weights):
        print(f"Asset: {asset:<6} | Targeted Allocation Weight: {weight*100:6.2f}%")
    print("-"*65)
    print(f"Expected Annualized Portfolio Return:     {p_ret*100:.2f}%")
    print(f"Expected Annualized Portfolio Volatility: {p_vol*100:.2f}%")
    print(f"Optimized Portfolio Sharpe Ratio:         {p_sharpe:.4f}")
    
    print("\n" + "="*65)
    print("📊 SECTION 3: HISTORICAL BACKTEST PERFORMANCE VS BENCHMARK (60/40)")
    print("="*65)
    print(f"Metric             | Optimized Strategy  | Institutional 60/40")
    print("-"*65)
    print(f"Total Growth (%)   | {strat_risk['Total Return']*100:18.2f}% | {bench_risk['Total Return']*100:18.2f}%")
    print(f"Max Drawdown (%)   | {strat_risk['Max Drawdown']*100:18.2f}% | {bench_risk['Max Drawdown']*100:18.2f}%")
    print(f"99% Daily VaR (%)  | {strat_risk['99% VaR']*100:18.2f}% | {bench_risk['99% VaR']*100:18.2f}%")
    print("="*65)
    print("\n✅ Entire multi-stage quantitative pipeline executed successfully.")

if __name__ == "__main__":
    execute_production_pipeline()