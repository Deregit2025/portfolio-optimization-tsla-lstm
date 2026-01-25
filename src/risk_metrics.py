import numpy as np

def calculate_var(df, confidence=0.05):
    """Value at Risk at specified confidence level"""
    return df["Return"].quantile(confidence)

def calculate_sharpe_ratio(df, risk_free_rate=0.01):
    """Annualized Sharpe Ratio"""
    mean_return = df["Return"].mean() * 252  # annualized
    vol = df["Return"].std() * np.sqrt(252)  # annualized
    sharpe = (mean_return - risk_free_rate) / vol
    return sharpe
