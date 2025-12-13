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
    best_e = float("inf")
    best_z = 0
    for z in range(1 << N):
        if bin(z).count("1") != K:
            continue
        e = float(energies[z])
        if e < best_e:
            best_e = e
            best_z = z
    return best_e, best_z
