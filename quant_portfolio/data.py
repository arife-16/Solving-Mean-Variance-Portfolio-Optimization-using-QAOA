import numpy as np
import random

def generate_synthetic_returns(N: int, T: int, seed: int):
    random.seed(seed)
    np.random.seed(seed)
    base = np.random.normal(0.0005, 0.01, size=(N, T))
    return base

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
