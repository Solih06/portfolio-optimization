import numpy as np
import pandas as pd
import scipy.optimize as sco

def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
    """
    Calculates annualized expected return, volatility, and Sharpe ratio for a set of asset weights.
    """
    # Annualize expected return (252 trading days)
    port_return = np.sum(mean_returns * weights) * 252
    # Annualize portfolio volatility
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    # Calculate Sharpe Ratio
    sharpe_ratio = (port_return - risk_free_rate) / port_volatility
    
    return port_return, port_volatility, sharpe_ratio

def optimize_max_sharpe(log_returns, risk_free_rate=0.02):
    """
    Uses numerical optimization to find the exact asset weights maximizing the Sharpe Ratio.
    """
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(log_returns.columns)
    
    # Negative Sharpe function to pass into the minimizer
    def negative_sharpe(weights):
        return -calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]
    
    # Constraints: The sum of all asset weights must equal 100% (1.0)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: Long-only portfolio (weights between 0% and 100% per asset)
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    # Equal weight initialization guess
    initial_guess = num_assets * [1.0 / num_assets]
    
    # Run Sequential Least Squares Programming (SLSQP) optimizer
    result = sco.minimize(negative_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if not result.success:
        raise RuntimeError("Portfolio optimization failed to converge.")
        
    optimized_weights = result.x
    return optimized_weights

def generate_efficient_frontier_data(log_returns, num_portfolios=500, risk_free_rate=0.02):
    """
    Simulates random allocations to visually map out the Risk vs. Return Efficient Frontier space.
    """
    mean_returns = log_returns.mean()
    cov_matrix = log_returns.cov()
    num_assets = len(log_returns.columns)
    
    results = np.zeros((3, num_portfolios))
    weights_record = []
    
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights) # Normalize to 1.0
        weights_record.append(weights)
        
        p_ret, p_vol, p_sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        results[0, i] = p_vol
        results[1, i] = p_ret
        results[2, i] = p_sharpe
        
    return results, weights_record