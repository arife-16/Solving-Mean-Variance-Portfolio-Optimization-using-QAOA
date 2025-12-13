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
