"""Real stock data loading - P1 contribution"""
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_real_data(tickers, start, end):
    """Fetch from Yahoo Finance"""
    print(f"Fetching {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, end=end, progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data
    print(f"✓ Downloaded {len(prices)} days")
    return prices

def compute_returns_from_prices(prices, method='log'):
    """Calculate returns"""
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = (prices / prices.shift(1) - 1)
    # Replace infinities with NaN, then drop rows with any NaNs to ensure aligned timepoints
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    return returns.values.T

def annualized_mu_sigma(returns, periods_per_year=252):
    """Annualize mu and sigma"""
    R = np.array(returns, dtype=float)
    # Clean NaN/Inf and clip extreme values to avoid overflow
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    R = np.clip(R, -1.0, 1.0)
    mu = R.mean(axis=1) * periods_per_year
    T = R.shape[1]
    if T <= 1:
        sigma = np.zeros((R.shape[0], R.shape[0]), dtype=float)
        return mu, sigma
    # Use np.cov for numerical stability
    sigma = np.cov(R, rowvar=True, bias=False) * periods_per_year
    # Final cleanup in case of numerical issues
    sigma = np.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    return mu, sigma
