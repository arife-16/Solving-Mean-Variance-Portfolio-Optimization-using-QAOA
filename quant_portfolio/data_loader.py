"""Real stock data loading - P1 contribution"""
import numpy as np
import pandas as pd
import yfinance as yf

def fetch_real_data(tickers, start, end):
    """Fetch stock price data from Yahoo Finance"""
    print(f"Fetching {len(tickers)} stocks...")
    data = yf.download(tickers, start=start, end=end, progress=False)
    
    # Extract only 'Adj Close' prices (not all OHLC data)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Adj Close']  # Get just adjusted close prices
    else:
        prices = data  # Single ticker case
    
    print(f"✓ Downloaded {len(prices)} days")
    return prices

def compute_returns_from_prices(prices, method='log'):
    """Calculate returns"""
    if method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    else:
        returns = (prices / prices.shift(1) - 1).dropna()
    
    result = returns.values
    print(f"DEBUG compute_returns: prices.shape={prices.shape}, result.shape={result.shape}")
    
    # If shape is wrong, transpose it
    if result.shape[0] < result.shape[1]:
        print(f"DEBUG: Transposing from {result.shape}")
        result = result.T
        print(f"DEBUG: After transpose: {result.shape}")
    
    return result

def annualized_mu_sigma(returns, periods_per_year=252):
    """Annualize mu and sigma"""
    # returns comes in as (T, N) from compute_returns_from_prices
    # We need (N, T) for formulations
    
    # Transpose to (N, T)
    if returns.shape[0] > returns.shape[1]:
        returns = returns.T
    
    mu = returns.mean(axis=1) * periods_per_year
    T = returns.shape[1]
    centered = returns - returns.mean(axis=1, keepdims=True)
    sigma = (centered @ centered.T) / (T - 1) * periods_per_year
    return mu, sigma, returns  # ← Return transposed returns too!