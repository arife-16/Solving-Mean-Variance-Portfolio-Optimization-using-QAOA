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

def energies_full(mu, sigma, q, N, K=None, penalty=100.0):
    # Vectorized implementation for speed
    n_states = 1 << N
    vals = np.zeros(n_states, dtype=float)
    indices = np.arange(n_states, dtype=np.int32) # int32 is enough for N=24 (16M)
    
    # Precompute bits to avoid repeated shifting
    bits = [(indices >> i) & 1 for i in range(N)]
    
    # Linear term: -mu @ x
    for i in range(N):
        vals -= mu[i] * bits[i]
        
    # Quadratic term: q * x @ sigma @ x
    # x^T S x = sum_i S_ii x_i^2 + 2 sum_{i<j} S_ij x_i x_j
    # Since x_i^2 = x_i for binary vars
    for i in range(N):
        vals += q * sigma[i, i] * bits[i]
        for j in range(i + 1, N):
            vals += 2 * q * sigma[i, j] * (bits[i] * bits[j])
            
    if K is not None:
        # Vectorized Hamming weight
        hamming = sum(bits)
        vals += penalty * (hamming - K)**2
        
    return vals

def energy_mad_bitstring(returns, q, x_bits):
    x = np.array(x_bits, dtype=float)
    k = max(int(x.sum()), 1)
    
    if returns.shape[0] < returns.shape[1]: 
        rp = (returns.T @ x) / k
    else:
        rp = (returns @ x) / k
    
    mu_p = rp.mean()
    mad = np.abs(rp - mu_p).mean()
    exp_ret = rp.mean()
    return float(q * mad - exp_ret)

def energy_mvo_tc_bitstring(mu, sigma, q, x_bits, tc, lam):
    x = np.array(x_bits, dtype=float)
    return float(q * x @ sigma @ x - mu @ x + lam * (tc @ x))

def energies_full_mad(returns, q, N, K=None, penalty=100.0):
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        val = energy_mad_bitstring(returns, q, b)
        if K is not None:
            card = sum(b)
            val += penalty * (card - K)**2
        vals[z] = val
    return vals

def energies_full_mvo_tc(mu, sigma, q, N, tc, lam, K=None, penalty=100.0):
    vals = np.zeros(1 << N, dtype=float)
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        val = energy_mvo_tc_bitstring(mu, sigma, q, b, tc, lam)
        if K is not None:
            card = sum(b)
            val += penalty * (card - K)**2
        vals[z] = val
    return vals
