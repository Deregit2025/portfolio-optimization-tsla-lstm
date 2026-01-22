import yfinance as yf
import pandas as pd
from pathlib import Path

def fetch_yfinance_data(tickers, start, end, save_path="data/raw"):
    """Download financial data and save CSVs"""
    Path(save_path).mkdir(parents=True, exist_ok=True)
    all_data = {}
    for ticker in tickers:
        df = yf.download(ticker, start=start, end=end)
        df.to_csv(f"{save_path}/{ticker}.csv")
        all_data[ticker] = df
    return all_data

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start = "2015-01-01"
    end = "2026-01-15"
    fetch_yfinance_data(tickers, start, end)
    print("Data fetched and saved to data/raw/")
