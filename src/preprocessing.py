# src/preprocessing.py
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import sys

# -----------------------------
# Functions
# -----------------------------

def load_and_clean(file_path):
    """
    Load CSV (with multi-row header), sort by date, fill missing values safely.
    
    Args:
        file_path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Cleaned dataframe with numeric columns
    """
    df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
    
    # Flatten multi-level columns: ("Price","TSLA") -> "Price"
    df.columns = [col[0] for col in df.columns]
    
    # Convert all columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='coerce')
        
    # Sort by date
    df = df.sort_index()

    # Forward fill missing values
    df = df.ffill()

    # Backfill in case NaNs at the top
    df = df.bfill()

    # Ensure no NaNs remain (optional)
    df = df.dropna(how="any")
    return df

def calculate_returns(df):
    """
    Calculate daily returns based on Close price.
    
    Args:
        df (pd.DataFrame): Dataframe containing 'Close'
    
    Returns:
        pd.DataFrame: Original dataframe with 'Return' column
    """
    df["Return"] = df["Close"].pct_change()
    return df

def rolling_volatility(df, window=20):
    """
    Calculate rolling volatility and add as a column.
    
    Args:
        df (pd.DataFrame)
        window (int): Rolling window size (days)
    
    Returns:
        pd.DataFrame
    """
    df["RollingVol"] = df["Return"].rolling(window=window).std()
    return df

def normalize_column(df, column, method="minmax"):
    """
    Normalize a column using Min-Max or Z-score scaling.
    
    Args:
        df (pd.DataFrame)
        column (str): Column to normalize
        method (str): "minmax" or "zscore"
    
    Returns:
        pd.Series: Normalized column
    """
    if method == "minmax":
        return (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    elif method == "zscore":
        return (df[column] - df[column].mean()) / df[column].std()
    else:
        raise ValueError("method must be 'minmax' or 'zscore'")

def detect_outliers(df, column, threshold=3):
    """
    Detect outliers based on Z-score.
    
    Args:
        df (pd.DataFrame)
        column (str)
        threshold (float): Z-score threshold
    
    Returns:
        pd.DataFrame: Boolean series where True indicates outlier
    """
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return abs(z_scores) > threshold

def plot_price(df, ticker):
    """Plot Close price over time"""
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label=f"{ticker} Close", color="blue")
    plt.title(f"{ticker} Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_returns(df, ticker):
    """Plot daily returns"""
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Return"], label=f"{ticker} Daily Returns", color="orange")
    plt.title(f"{ticker} Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rolling_volatility(df, ticker):
    """Plot rolling volatility"""
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["RollingVol"], label=f"{ticker} Rolling Volatility (20 days)", color="green")
    plt.title(f"{ticker} Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.grid(True)
    plt.show()

def check_stationarity(series, ticker="", column_name=""):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    print(f"--- {ticker} ADF Test for {column_name} ---")
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
    df = df.dropna(subset=["Return", "RollingVol"])

    # Outlier detection
    outliers = detect_outliers(df, "Return")
    print(f"{outliers.sum()} outliers detected in daily returns.")

    # Stationarity check
    check_stationarity(df["Return"], ticker, "Return")
    check_stationarity(df["Close"], ticker, "Close")

    # Risk metrics
    var_95 = calculate_var(df)
    sharpe = calculate_sharpe_ratio(df)
    print(f"{ticker} VaR (5%): {var_95:.4f}")
    print(f"{ticker} Sharpe Ratio: {sharpe:.4f}\n")

    # Plotting (feedback requested explicit plots)
    plot_price(df, ticker)
    plot_returns(df, ticker)
    plot_rolling_volatility(df, ticker)

    # Save processed data
    save_processed(df, ticker)
