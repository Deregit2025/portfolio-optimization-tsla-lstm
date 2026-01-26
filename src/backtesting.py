# src/backtesting.py

import numpy as np
import pandas as pd


def compute_portfolio_returns(returns_df: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute daily portfolio returns.
    
    returns_df: DataFrame with columns as assets and rows as daily returns
    weights: dict {asset: weight}
    """
    w = np.array([weights[col] for col in returns_df.columns])
    portfolio_returns = returns_df.values @ w
    return pd.Series(portfolio_returns, index=returns_df.index)


def cumulative_returns(daily_returns: pd.Series) -> pd.Series:
    return (1 + daily_returns).cumprod() - 1


def total_return(daily_returns: pd.Series) -> float:
    return (1 + daily_returns).prod() - 1


def annualized_return(daily_returns: pd.Series, periods: int = 252) -> float:
    tr = total_return(daily_returns)
    n_days = len(daily_returns)
    return (1 + tr) ** (periods / n_days) - 1


def sharpe_ratio(daily_returns: pd.Series, periods: int = 252) -> float:
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(periods)


def max_drawdown(cumulative_returns: pd.Series) -> float:
    wealth = 1 + cumulative_returns
    running_max = wealth.cummax()
    drawdown = wealth / running_max - 1
    return drawdown.min()
