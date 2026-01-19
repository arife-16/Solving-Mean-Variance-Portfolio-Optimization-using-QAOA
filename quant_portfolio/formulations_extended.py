"""CVaR risk formulation - P1 contribution"""
import numpy as np

def energy_cvar_bitstring(returns, q, x_bits, alpha=0.05):
    """CVaR risk measure (tail risk focus)"""
    x = np.array(x_bits, dtype=float)
    k = max(int(x.sum()), 1)
    
    portfolio_returns = (returns.T @ x) / k
    mean_return = portfolio_returns.mean()
    
    cutoff_idx = max(1, int(alpha * len(portfolio_returns)))
    worst_returns = np.sort(portfolio_returns)[:cutoff_idx]
    cvar = worst_returns.mean()
    
    return float(q * (-cvar) - mean_return)

def energies_full_cvar(returns, q, N, alpha=0.05):
    """Compute CVaR for all 2^N configurations"""
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        vals[z] = energy_cvar_bitstring(returns, q, b, alpha)
    return vals

def energies_k_hot_cvar(returns, q, N, K, alpha=0.05):
    """CVaR for K-hot subspace only"""
    vals = []
    idxs = []
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        if sum(b) == K:
            vals.append(energy_cvar_bitstring(returns, q, b, alpha))
            idxs.append(z)
    return np.array(vals), np.array(idxs)