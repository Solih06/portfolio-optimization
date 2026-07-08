import numpy as np
import pandas as pd

def run_historical_backtest(log_returns, strategy_weights, benchmark_weights=None):
    """
    Simulates historical equity growth for the optimized strategy vs a 60/40 benchmark.
    """
    if benchmark_weights is None:
        # Standard institutional 60/40 allocation: 0% TSLA, 60% SPY, 40% BND
        benchmark_weights = np.array([0.0, 0.60, 0.40])
        
    # Calculate daily returns for both portfolios
    strategy_daily_ret = log_returns.dot(strategy_weights)
    benchmark_daily_ret = log_returns.dot(benchmark_weights)
    
    # Convert log returns back to simple returns to compound wealth accurately
    strat_simple_ret = np.exp(strategy_daily_ret) - 1
    bench_simple_ret = np.exp(benchmark_daily_ret) - 1
    
    # Generate equity curves assuming an initial $10,000 investment
    initial_capital = 10000.0
    strategy_equity = initial_capital * (1 + strat_simple_ret).cumprod()
    benchmark_equity = initial_capital * (1 + bench_simple_ret).cumprod()
    
    return strategy_equity, benchmark_equity, strat_simple_ret, bench_simple_ret

def compute_risk_metrics(equity_curve, daily_simple_returns):
    """
    Calculates total return, maximum drawdown, and historical 99% Value at Risk (VaR).
    """
    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    
    # Maximum Drawdown (Peak to Trough drop)
    running_max = equity_curve.cummax()
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = drawdowns.min()
    
    # 99% Historical Value at Risk (VaR)
    # Represents the maximum percentage loss expected over a single day with 99% confidence
    var_99 = np.percentile(daily_simple_returns, 1)
    
    return {
        "Total Return": total_return,
        "Max Drawdown": max_drawdown,
        "99% VaR": var_99
    }