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


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a raw CSV file with Date as index.
    Used by tests and preprocessing.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    df = pd.read_csv(
        file_path,
        index_col=0,
        parse_dates=True
    )
    return df


if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start = "2015-01-01"
    end = "2026-01-15"
    fetch_yfinance_data(tickers, start, end)
    print("Data fetched and saved to data/raw/")
