import numpy as np
import random

def generate_synthetic_returns(N: int, T: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # Generate correlated assets (market factor + sector factors + idiosyncratic)
    # 1. Market mode
    mkt = np.random.normal(0.0005, 0.01, size=T)
    # 2. Random sectors (e.g., N/2 assets per sector)
    n_sectors = max(1, N // 4)
    sectors = []
    for _ in range(n_sectors):
        sectors.append(np.random.normal(0.0, 0.005, size=T))
    
    returns = np.zeros((N, T))
    for i in range(N):
        # Assign to a random sector
        sec_idx = i % n_sectors
        # Idiosyncratic
        idio = np.random.normal(0.0, 0.015, size=T)
        # Beta to market and sector
        beta_mkt = np.random.uniform(0.5, 1.5)
        beta_sec = np.random.uniform(0.5, 1.5)
        
        returns[i] = beta_mkt * mkt + beta_sec * sectors[sec_idx] + idio
        
    return returns

def compute_mu_sigma(returns: np.ndarray):
    mu = returns.mean(axis=1)
    X = returns - mu[:, None]
    sigma = (X @ X.T) / (returns.shape[1] - 1)
    return mu, sigma

def generate_transaction_costs(N: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return np.abs(np.random.normal(0.001, 0.0005, size=N))

def load_prices_csv(path: str):
    a = np.genfromtxt(path, delimiter=",")
    return a

def returns_from_prices(prices: np.ndarray):
    r = prices[1:] / np.maximum(prices[:-1], 1e-12) - 1.0
    return r.T

def load_transaction_costs_csv(path: str):
    v = np.genfromtxt(path, delimiter=",")
    return v
