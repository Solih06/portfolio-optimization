import yfinance as yf
import pandas as pd

def fetch_raw_data(tickers, start_date, end_date):
    """
    Ingests daily market close prices from Yahoo Finance.
    """
    try:
        print(f"🔬 Ingesting raw market feeds for {tickers} from {start_date} to {end_date}...")
        # Download data (auto_adjust=True is active by default)
        raw_data = yf.download(tickers, start=start_date, end=end_date)['Close']
        return raw_data
    except Exception as e:
        raise RuntimeError(f"Data ingestion failed: {e}")

def clean_market_data(df):
    """
    Applies forward-fill and backward-fill to handle missing market data points.
    """
    return df.ffill().bfill()

def transform_to_log_returns(df):
    """
    Transforms price series data into stationary log returns.
    """
    import numpy as np
    return np.log(df / df.shift(1)).dropna()