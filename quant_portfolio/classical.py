import numpy as np

def brute_force_k_hot(mu, sigma, q, N, K):
    best_e = float("inf")
    best_z = 0
    for z in range(1 << N):
        b = [(z >> i) & 1 for i in range(N)]
        if sum(b) != K:
            continue
        x = np.array(b, dtype=float)
        e = float(q * x @ sigma @ x - mu @ x)
        if e < best_e:
            best_e = e
            best_z = z
    return best_e, best_z

def local_search(mu, sigma, q, N, K, z_init):
    z = z_init
    improved = True
    while improved:
        improved = False
        ones = [i for i in range(N) if (z >> i) & 1]
        zeros = [i for i in range(N) if ((z >> i) & 1) == 0]
        for i in ones:
            for j in zeros:
                z2 = z ^ (1 << i) ^ (1 << j)
                b = [(z2 >> k) & 1 for k in range(N)]
                x = np.array(b, dtype=float)
                e = float(q * x @ sigma @ x - mu @ x)
                b0 = [(z >> k) & 1 for k in range(N)]
                x0 = np.array(b0, dtype=float)
                e0 = float(q * x0 @ sigma @ x0 - mu @ x0)
                if e < e0:
                    z = z2
                    improved = True
                    break
            if improved:
                break
    return z

def brute_force_from_energies(energies: np.ndarray, N: int, K: int):
    # Vectorized search for optimal energy with Hamming weight K
    # Much faster than iterating in Python for large N
    
    # 1. Generate Hamming weights efficiently
    # For N=24, generating 16M indices is fine (128MB for int64, 64MB for int32)
    indices = np.arange(len(energies), dtype=np.int32)
    
    # Vectorized population count (Hamming weight)
    # popcount(i) = i - (i >> 1) & 0x5555...
    # This is a bit complex to implement generically for N bits in numpy without bitwise magic.
    # Simpler: sum bits
    hamming = np.zeros(len(energies), dtype=np.int8)
    for i in range(N):
        hamming += ((indices >> i) & 1).astype(np.int8)
        
    # 2. Filter for K-hot states
    mask = (hamming == K)
    
    # 3. Find min energy in masked array
    # We set non-K states to infinity to ignore them
    # But energies might be modified, better to just slice
    valid_energies = energies[mask]
    valid_indices = indices[mask]
    
    if len(valid_energies) == 0:
        return float("inf"), 0
        
    min_idx_local = np.argmin(valid_energies)
    best_e = float(valid_energies[min_idx_local])
    best_z = int(valid_indices[min_idx_local])
    
    return best_e, best_z
