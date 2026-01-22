# src/preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
import sys

# -----------------------------
# Functions
# -----------------------------

def load_and_clean(file_path):
    """
    Load CSV (with multi-row header), sort by date, fill missing values safely
    """
    # Load CSV with first two rows as header
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # Flatten multi-level columns: ("Price","TSLA") -> "Price"
    df.columns = [col[0] for col in df.columns]
    
    # Convert all columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Sort by date
    df = df.sort_index()
    
    # Forward fill missing values
    df = df.ffill()  # safer alternative to df.fillna(method="ffill")
    
    return df



def calculate_returns(df):
    """Calculate daily returns based on Close price"""
    # Use 'Close' column instead of 'Adj Close'
    df["Return"] = df["Close"].pct_change()
    return df


def rolling_volatility(df, window=20):
    """Add rolling volatility column"""
    df["RollingVol"] = df["Return"].rolling(window=window).std()
    return df

def check_stationarity(series, ticker=""):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    print(f"--- {ticker} ADF Test ---")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("Series is stationary ✅\n")
    else:
        print("Series is non-stationary ❌\n")

def calculate_var(df, confidence=0.05):
    """Value at Risk at specified confidence level"""
    return df["Return"].quantile(confidence)

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    """Annualized Sharpe Ratio"""
    mean_return = df["Return"].mean() * 252
    vol = df["Return"].std() * np.sqrt(252)
    sharpe = (mean_return - risk_free_rate) / vol
    return sharpe

def save_processed(df, ticker, save_path="data/processed"):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{save_path}/{ticker}_cleaned.csv")
    print(f"{ticker} cleaned data saved at {save_path}/{ticker}_cleaned.csv\n")

# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide ticker symbol, e.g.: python -m src.preprocessing TSLA")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    raw_file = f"data/raw/{ticker}.csv"

    # Load & clean
    df = load_and_clean(raw_file)

    # Calculate returns & rolling volatility
    df = calculate_returns(df)
    df = rolling_volatility(df)

    # Stationarity check
    check_stationarity(df["Return"], ticker)
    check_stationarity(df["Close"], ticker)

    # Risk metrics
    var_95 = calculate_var(df)
    sharpe = calculate_sharpe_ratio(df)
    print(f"{ticker} VaR (5%): {var_95:.4f}")
    print(f"{ticker} Sharpe Ratio: {sharpe:.4f}\n")

    # Save processed data
    save_processed(df, ticker)
