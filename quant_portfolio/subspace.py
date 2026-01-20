import numpy as np
from scipy.special import comb

def generate_basis(N, K):
    """
    Generates all N-bit integers with Hamming weight K, sorted.
    Returns a numpy array of shape (binom(N, K),) with dtype int64.
    """
    # Base cases
    if K < 0 or K > N:
        return np.array([], dtype=np.int64)
    if K == 0:
        return np.array([0], dtype=np.int64)
    if K == N:
        return np.array([(1 << N) - 1], dtype=np.int64)
    
    # For N=30, K=15, size is ~155 million integers (1.2GB)
    count = comb(N, K, exact=True)
        
    # Logic:
    # Elements with MSB (bit N-1) = 0: generate_basis(N-1, K)
    # Elements with MSB (bit N-1) = 1: generate_basis(N-1, K-1) + 2^(N-1)
    
    # To save memory, we can preallocate the array and fill it
    basis = np.empty(count, dtype=np.int64)
    
    # Iterative approach to avoid deep recursion stack overhead with large arrays
    
    _fill_basis(basis, 0, N, K, 0)
    return basis

def _fill_basis(arr, start_idx, n, k, prefix):
    """
    Fills arr starting at start_idx with integers of n bits, weight k, ORed with prefix.
    Returns the next start_idx.
    """
    if k == 0:
        arr[start_idx] = prefix
        return start_idx + 1
    if k == n:
        arr[start_idx] = prefix | ((1 << n) - 1)
        return start_idx + 1
    
    # Count for MSB=0 branch (n-1, k)
    c0 = comb(n - 1, k, exact=True)
    
    # Fill MSB=0 branch
    idx = _fill_basis(arr, start_idx, n - 1, k, prefix)
    
    # Fill MSB=1 branch (n-1, k-1)
    # Bit at n-1 is set.
    new_prefix = prefix | (1 << (n - 1))
    return _fill_basis(arr, idx, n - 1, k - 1, new_prefix)

def compute_energies_subspace(states, mu, sigma, q, N, penalty=100.0):
    """
    Computes portfolio energy for each state in the subspace.
    E = q * x^T S x - mu^T x
    Since all states have weight K, no penalty term is needed (or it's constant 0).
    """
    n_samples = len(states)
    vals = np.zeros(n_samples, dtype=float)
    
    # Vectorized bit extraction?
    # For large N, extracting bits into (N, Samples) bool array is huge (30 * 155M bytes ~ 4.5GB).
    # We should iterate over bits instead.
    
    # Linear term: -mu @ x
    # Iterate over assets i
    for i in range(N):
        # Identify states where bit i is set
        mask = (1 << i)
        has_bit = (states & mask) != 0
        vals[has_bit] -= mu[i]
        
    # Quadratic term: q * x @ sigma @ x
    # sum_i S_ii x_i + 2 sum_{i<j} S_ij x_i x_j
    
    for i in range(N):
        # Diagonal term
        mask_i = (1 << i)
        has_bit_i = (states & mask_i) != 0
        vals[has_bit_i] += q * sigma[i, i]
        
        # Off-diagonal
        for j in range(i + 1, N):
            term = 2 * q * sigma[i, j]
            mask_j = (1 << j)
            # Both bits set
            mask_both = mask_i | mask_j
            has_both = (states & mask_both) == mask_both
            vals[has_both] += term
            
    return vals

def qaoa_expectation_subspace(states, energies, N, K, theta, mixer='xy', T=1):
    """
    Computes QAOA expectation value in the subspace.
    """
    g, b = np.split(theta, 2)
    dim = len(states)
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for t in range(len(g)):
        # Phase separator
        psi *= np.exp(-1j * g[t] * energies)
        
        # Mixer
        if mixer == 'xy':
            # Match qaoa_core.apply_xy_ring logic
            beta_step = 4 * b[t] / T
            
            for _ in range(T):
                # Layer 1: (0,1), (2,3), ...
                pairs_1 = [(i, i + 1) for i in range(0, N - 1, 2)]
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, pairs_1)
                
                # Layer 2: (1,2), (3,4), ...
                pairs_2 = [(i, i + 1) for i in range(1, N - 1, 2)]
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, pairs_2)
                
                # Layer 3: (N-1, 0)
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, [(N - 1, 0)])
                
    probs = np.real(psi.conj() * psi)
    return float(np.sum(probs * energies))

def evolve_state_subspace(states, energies, N, theta, mixer='xy', T=1):
    """
    Evolves the state in the subspace and returns the final state vector.
    """
    g, b = np.split(theta, 2)
    dim = len(states)
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    
    for t in range(len(g)):
        # Phase separator
        psi *= np.exp(-1j * g[t] * energies)
        
        # Mixer
        if mixer == 'xy':
            beta_step = 4 * b[t] / T
            for _ in range(T):
                pairs_1 = [(i, i + 1) for i in range(0, N - 1, 2)]
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, pairs_1)
                pairs_2 = [(i, i + 1) for i in range(1, N - 1, 2)]
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, pairs_2)
                psi = apply_xy_mixer_subspace(psi, states, beta_step, N, [(N - 1, 0)])
                
    return psi

def compute_overlap_subspace(psi, states, z_opt_val):
    """
    Computes overlap with the optimal solution.
    z_opt_val is the integer representation of the optimal state.
    """
    # Find index of z_opt_val in states
    idx = np.searchsorted(states, z_opt_val)
    if idx < len(states) and states[idx] == z_opt_val:
        return float(np.abs(psi[idx])**2)
    return 0.0

def apply_phase_separator_subspace(psi, energies, gamma):
    """
    Applies exp(-i * gamma * H_cost) to psi.
    """
    phase = np.exp(-1j * gamma * energies)
    return psi * phase

def apply_xy_mixer_subspace(psi, states, beta, N, pairs_list):
    """
    Applies XY mixer on subspace.
    U = exp(-i * beta * (XX + YY))
    For each pair (i,j), restricted to subspace {01, 10}, it is a rotation.
    |01> -> cos(2b)|01> - i sin(2b)|10>
    |10> -> cos(2b)|10> - i sin(2b)|01>
    """
    c = np.cos(2 * beta)
    s = np.sin(2 * beta)
    
    # For each pair in the ring/graph
    for (i, j) in pairs_list:
        mask_i = (1 << i)
        mask_j = (1 << j)
        mask_pair = mask_i | mask_j
        
        # We need to find states with exactly one of i, j set.
        # (states & mask_pair) must be mask_i or mask_j
        # But we only need to find one side (e.g., 01) and find its partner (10).        
        candidates = states & mask_pair
        
        # Indices where bit i is 1 and j is 0
        idx_10 = np.where(candidates == mask_i)[0]
        
        if len(idx_10) == 0:
            continue
            
        states_10 = states[idx_10]
        
        # Calculate partner states: flip i and j
        # 1 at i, 0 at j -> 0 at i, 1 at j
        states_01 = states_10 ^ mask_pair
        
        # Find indices of partner states
        # Since states is sorted, use searchsorted
        idx_01 = np.searchsorted(states, states_01)
        
        # Extract amplitudes
        psi_10 = psi[idx_10]
        psi_01 = psi[idx_01]
        
        # Update
        # |10>_new = c|10> - i*s|01>
        # |01>_new = c|01> - i*s|10>
        
        psi[idx_10] = c * psi_10 - 1j * s * psi_01
        psi[idx_01] = c * psi_01 - 1j * s * psi_10
        
    return psi
