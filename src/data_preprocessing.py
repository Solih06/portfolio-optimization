import numpy as np
import pandas as pd
import yfinance as yf

def fetch_raw_data(tickers=["TSLA", "BND", "SPY"], start_date="2015-01-01", end_date="2026-06-30"):
    """
    Fetches raw historical closing data directly from Yahoo Finance API.
    """
    print(f"🔬 Ingesting raw market feeds for {tickers} from {start_date} to {end_date}...")
    try:
        raw_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame(name=tickers[0])
        return raw_data
    except Exception as e:
        raise RuntimeError(f"Data ingestion failed: {e}")

def clean_market_data(df):
    """
    Cleans structural data alignment flaws using sequential forward and backward fills.
    """
    if df.isnull().sum().sum() > 0:
        print("⚠️ Gaps detected in trading timeline. Executing ffill/bfill cleaning sweep...")
        df = df.ffill().bfill()
    return df

def transform_to_log_returns(df):
    """
    Transforms non-stationary price levels into stationary daily percentage log returns.
    """
    log_returns = np.log(df / df.shift(1)).dropna()
    return log_returns

def extract_rolling_features(df, window=20):
    """
    Extracts rolling window means and standard deviations (volatility tracking).
    """
    rolling_mean = df.rolling(window=window).mean()
    rolling_std = df.rolling(window=window).std()
    return rolling_mean, rolling_std

def isolate_anomalous_outliers(returns_df, window=20, threshold=3.0):
    """
    Flags structural return outliers where daily return exceeds +/- 3 rolling standard deviations.
    """
    outliers_dict = {}
    for col in returns_df.columns:
        rolling_m = returns_df[col].rolling(window=window).mean()
        rolling_s = returns_df[col].rolling(window=window).std()
        z_scores = (returns_df[col] - rolling_m) / rolling_s
        outliers = returns_df[z_scores.abs() > threshold][col]
        outliers_dict[col] = outliers
    return outliers_dict