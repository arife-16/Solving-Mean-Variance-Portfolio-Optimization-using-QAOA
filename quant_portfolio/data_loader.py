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
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = (prices / prices.shift(1) - 1).dropna()
    return returns.values.T

def annualized_mu_sigma(returns, periods_per_year=252):
    """Annualize mu and sigma"""
    mu = returns.mean(axis=1) * periods_per_year
    T = returns.shape[1]
    centered = returns - returns.mean(axis=1, keepdims=True)
    sigma = (centered @ centered.T) / (T - 1) * periods_per_year
    return mu, sigma