# src/portfolio.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# 1. Compute Covariance Matrix
# -----------------------------
def compute_covariance_matrix(returns_df, annualize=True):
    """
    Compute the covariance matrix of asset returns.

    Args:
        returns_df (pd.DataFrame): Columns are asset returns.
        annualize (bool): If True, annualize the covariance (assume 252 trading days).

    Returns:
        pd.DataFrame: Covariance matrix
    """
    cov = returns_df.cov()
    if annualize:
        cov *= 252
    return cov

# -----------------------------
# 2. Portfolio Metrics
# -----------------------------
def portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate=0.0):
    """
    Calculate expected return, volatility, and Sharpe ratio for a portfolio.

    Args:
        weights (np.array): Portfolio weights
        expected_returns (np.array): Expected returns vector
        cov_matrix (pd.DataFrame or np.array): Covariance matrix
        risk_free_rate (float): Risk-free rate for Sharpe calculation

    Returns:
        tuple: (expected_return, volatility, sharpe_ratio)
    """
    weights = np.array(weights)
    port_return = np.dot(weights, expected_returns)
    port_vol = np.sqrt(weights.T @ cov_matrix @ weights)
    sharpe_ratio = (port_return - risk_free_rate) / port_vol if port_vol != 0 else 0
    return port_return, port_vol, sharpe_ratio

# -----------------------------
# 3. Optimize Portfolio
# -----------------------------
def optimize_portfolio(expected_returns, cov_matrix, risk_free_rate=0.0, 
                       method='sharpe', bounds=(0,1)):
    """
    Optimize portfolio weights to maximize Sharpe ratio or minimize volatility.

    Args:
        expected_returns (np.array): Expected returns vector
        cov_matrix (pd.DataFrame or np.array): Covariance matrix
        risk_free_rate (float): Risk-free rate
        method (str): 'sharpe' for max Sharpe, 'vol' for min volatility
        bounds (tuple): Bounds for each weight (min, max)

    Returns:
        dict: {'weights': np.array, 'return': float, 'volatility': float, 'sharpe': float}
    """
    n = len(expected_returns)
    bounds = tuple([bounds] * n)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # sum(weights)=1

    if method == 'sharpe':
        # Maximize Sharpe ratio (minimize negative)
        def neg_sharpe(weights):
            _, _, sharpe = portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)
            return -sharpe
        result = minimize(neg_sharpe, n*[1./n,], method='SLSQP', bounds=bounds, constraints=constraints)
    elif method == 'vol':
        # Minimize volatility
        def port_vol(weights):
            _, vol, _ = portfolio_metrics(weights, expected_returns, cov_matrix, risk_free_rate)
            return vol
        result = minimize(port_vol, n*[1./n,], method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        raise ValueError("Method must be 'sharpe' or 'vol'")

    w_opt = result.x
    r, v, s = portfolio_metrics(w_opt, expected_returns, cov_matrix, risk_free_rate)
    return {'weights': w_opt, 'return': r, 'volatility': v, 'sharpe': s}

# -----------------------------
# 4. Generate Efficient Frontier
# -----------------------------
def efficient_frontier(expected_returns, cov_matrix, points=50, risk_free_rate=0.0):
    """
    Generate the efficient frontier for given assets.

    Args:
        expected_returns (np.array): Expected returns vector
        cov_matrix (pd.DataFrame or np.array): Covariance matrix
        points (int): Number of portfolios to simulate along the frontier
        risk_free_rate (float): Risk-free rate

    Returns:
        tuple: (volatilities, returns, sharpe_ratios, weights_list)
    """
    n = len(expected_returns)
    volatilities = []
    returns = []
    sharpe_ratios = []
    weights_list = []

    for target_return in np.linspace(min(expected_returns), max(expected_returns), points):
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}
        )
        bounds = tuple([(0, 1) for _ in range(n)])
        # minimize volatility for given target return
        result = minimize(lambda w: np.sqrt(w.T @ cov_matrix @ w),
                          n*[1./n,],
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
        if result.success:
            w = result.x
            r, v, s = portfolio_metrics(w, expected_returns, cov_matrix, risk_free_rate)
            volatilities.append(v)
            returns.append(r)
            sharpe_ratios.append(s)
            weights_list.append(w)

    return np.array(volatilities), np.array(returns), np.array(sharpe_ratios), weights_list

# -----------------------------
# 5. Plot Efficient Frontier
# -----------------------------
def plot_efficient_frontier(volatilities, returns, sharpe_ratios, 
                            max_sharpe_port=None, min_vol_port=None, save_path=None):
    """
    Plot the efficient frontier with key portfolios marked.

    Args:
        volatilities (array): Portfolio volatilities
        returns (array): Portfolio returns
        sharpe_ratios (array): Portfolio Sharpe ratios
        max_sharpe_port (dict): Portfolio dict with max Sharpe
        min_vol_port (dict): Portfolio dict with min volatility
        save_path (str): Path to save figure
    """
    plt.figure(figsize=(10,6))
    plt.plot(volatilities, returns, 'b--', label='Efficient Frontier')
    if max_sharpe_port:
        plt.scatter(max_sharpe_port['volatility'], max_sharpe_port['return'], 
                    c='r', marker='*', s=200, label='Max Sharpe')
    if min_vol_port:
        plt.scatter(min_vol_port['volatility'], min_vol_port['return'], 
                    c='g', marker='*', s=200, label='Min Volatility')
    plt.xlabel('Volatility (Std Dev)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.legend()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

