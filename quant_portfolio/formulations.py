import numpy as np

def energy_mvo_bitstring(mu, sigma, q, x_bits):
    x = np.array(x_bits, dtype=float)
    return float(q * x @ sigma @ x - mu @ x)

def energies_k_hot(mu, sigma, q, N, K):
    vals = []
    idxs = []
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        if sum(b) == K:
            vals.append(energy_mvo_bitstring(mu, sigma, q, b))
            idxs.append(z)
    return np.array(vals), np.array(idxs)

def energies_full(mu, sigma, q, N):
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        vals[z] = energy_mvo_bitstring(mu, sigma, q, b)
    return vals

def energy_mad_bitstring(returns, q, x_bits):
    x = np.array(x_bits, dtype=float)
    k = max(int(x.sum()), 1)
    rp = (returns.T @ x) / k
    mu_p = rp.mean()
    mad = np.abs(rp - mu_p).mean()
    exp_ret = rp.mean()
    return float(q * mad - exp_ret)

def energy_mvo_tc_bitstring(mu, sigma, q, x_bits, tc, lam):
    x = np.array(x_bits, dtype=float)
    return float(q * x @ sigma @ x - mu @ x + lam * (tc @ x))

def energies_full_mad(returns, q, N):
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        vals[z] = energy_mad_bitstring(returns, q, b)
    return vals

def energies_full_mvo_tc(mu, sigma, q, N, tc, lam):
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        vals[z] = energy_mvo_tc_bitstring(mu, sigma, q, b, tc, lam)
    return vals
