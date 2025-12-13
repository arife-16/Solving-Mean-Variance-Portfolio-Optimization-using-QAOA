import numpy as np

def dicke_state(N: int, K: int):
    dim = 1 << N
    psi = np.zeros(dim, dtype=complex)
    c = 0
    for z in range(dim):
        if bin(z).count("1") == K:
            psi[z] = 1.0
            c += 1
    if c == 0:
        psi[0] = 1.0
        return psi
    return psi / np.sqrt(c)

def warm_start_state(mu, sigma, K):
    N = mu.shape[0]
    p = 1 / (1 + np.exp(-mu))
    w = np.array([np.log(p[i] / (1 - p[i])) for i in range(N)])
    dim = 1 << N
    amp = np.zeros(dim, dtype=float)
    for z in range(dim):
        if bin(z).count("1") == K:
            s = 0.0
            for i in range(N):
                if (z >> i) & 1:
                    s += w[i]
            amp[z] = np.exp(s)
    if amp.sum() == 0:
        return dicke_state(N, K)
    psi = amp / np.linalg.norm(amp)
    return psi.astype(complex)

def rx(theta):
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=complex)

def xy_unitary(beta):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    H = np.kron(X, X) + np.kron(Y, Y)
    vals, vecs = np.linalg.eigh(H)
    U = (vecs @ np.diag(np.exp(-1j * beta * vals)) @ vecs.conj().T)
    return U

def apply_xy_pair(state, beta, i, j, N, T):
    U = xy_unitary(4 * beta / T)
    psi = state
    for _ in range(T):
        psi = apply_two_qubit(psi, U, i, j, N)
    return psi

def apply_two_qubit(state, U, i, j, N):
    order = list(range(N))
    if i > j:
        i, j = j, i
    psi = state.reshape([2] * N)
    psi = np.moveaxis(psi, [i, j], [0, 1])
    s = psi.reshape(4, -1)
    s = U @ s
    psi = s.reshape([2, 2] + [2] * (N - 2))
    psi = np.moveaxis(psi, [0, 1], [i, j])
    return psi.reshape(1 << N)

def apply_rx_all(state, beta, N):
    psi = state.reshape([2] * N)
    U = rx(2 * beta)
    for i in range(N):
        psi = np.tensordot(U, psi, axes=[[1], [i]])
    return psi.reshape(1 << N)

def apply_xy_ring(state, beta, N, T):
    U = xy_unitary(4 * beta / T)
    psi = state
    for _ in range(T):
        for i in range(0, N - 1, 2):
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        for i in range(1, N - 1, 2):
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        psi = apply_two_qubit(psi, U, N - 1, 0, N)
    return psi

def phase_separator(state, energies, theta):
    g, b = np.split(theta, 2)
    psi = state.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        if b[t] == 0:
            continue
    return psi

def qaoa_layer(state, energies, N, beta, mixer, T):
    if mixer == "x":
        return apply_rx_all(state, beta, N)
    return apply_xy_ring(state, beta, N, T)

def qaoa_expectation_ops(psi0, energies, N, theta, ops, T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        kind, info = ops[t]
        if kind == "x":
            psi = apply_rx_all(psi, b[t], N)
        elif kind == "xy_ring":
            psi = apply_xy_ring(psi, b[t], N, T)
        else:
            i, j = info
            psi = apply_xy_pair(psi, b[t], i, j, N, T)
    probs = np.real(psi.conj() * psi)
    return float((probs * energies).sum())

def qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        kind, info = ops[t]
        if kind == "x":
            psi = apply_rx_all(psi, b[t], N)
        elif kind == "xy_ring":
            psi = apply_xy_ring(psi, b[t], N, T)
        else:
            i, j = info
            psi = apply_xy_pair(psi, b[t], i, j, N, T)
    probs = np.real(psi.conj() * psi)
    idx = np.argsort(energies)[::-1]
    cum = np.cumsum(probs[idx])
    thr = alpha
    s = 0.0
    w = 0.0
    for k in range(len(idx)):
        if w >= thr:
            break
        take = min(thr - w, probs[idx[k]])
        s += energies[idx[k]] * take
        w += take
    return float(s / max(thr, 1e-8))

def apply_depolarizing(probs, p):
    d = probs.shape[0]
    return (1.0 - p) * probs + p * (1.0 / d)

def qaoa_expectation_shots(psi0, energies, N, K, theta, mixer="xy", T=1, shots=1024, noise_p=0.0):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T)
    probs = np.real(psi.conj() * psi)
    if noise_p > 0.0:
        probs = apply_depolarizing(probs, noise_p)
    probs = probs / probs.sum()
    idx = np.arange(probs.shape[0])
    counts = np.random.multinomial(shots, probs)
    est = float((counts * energies).sum() / max(shots, 1))
    return est

def qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer="xy", T=1, shots=1024, noise_p=0.0):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T)
    probs = np.real(psi.conj() * psi)
    if noise_p > 0.0:
        probs = apply_depolarizing(probs, noise_p)
    probs = probs / probs.sum()
    idx = np.arange(probs.shape[0])
    counts = np.random.multinomial(shots, probs)
    order = np.argsort(energies)[::-1]
    cum = np.cumsum(counts[order])
    thr = int(alpha * shots)
    s = 0.0
    w = 0
    for k in range(len(order)):
        if w >= thr:
            break
        take = min(thr - w, counts[order[k]])
        s += energies[order[k]] * take
        w += take
    return float(s / max(thr, 1))

def evolve_state(psi0, energies, N, theta, mixer="xy", T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T)
    return psi

def evolve_state_ops(psi0, energies, N, theta, ops, T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        kind, info = ops[t]
        if kind == "x":
            psi = apply_rx_all(psi, b[t], N)
        elif kind == "xy_ring":
            psi = apply_xy_ring(psi, b[t], N, T)
        else:
            i, j = info
            psi = apply_xy_pair(psi, b[t], i, j, N, T)
    return psi

def compute_overlap(psi, z_opt, noise_p=0.0, shots=0):
    probs = np.real(psi.conj() * psi)
    if noise_p > 0.0:
        probs = apply_depolarizing(probs, noise_p)
    probs = probs / max(probs.sum(), 1e-12)
    if shots and shots > 0:
        counts = np.random.multinomial(shots, probs)
        return float(counts[z_opt] / max(shots, 1))
    return float(probs[z_opt])

def qaoa_expectation(psi0, energies, N, K, theta, mixer="xy", T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T)
    probs = np.real(psi.conj() * psi)
    return float((probs * energies).sum())

def qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer="xy", T=1):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T)
    probs = np.real(psi.conj() * psi)
    idx = np.argsort(energies)[::-1]
    cum = np.cumsum(probs[idx])
    thr = alpha
    s = 0.0
    w = 0.0
    for k in range(len(idx)):
        if w >= thr:
            break
        take = min(thr - w, probs[idx[k]])
        s += energies[idx[k]] * take
        w += take
    return float(s / max(thr, 1e-8))

def gate_counts(N, p, mixer, T):
    if mixer == "x":
        return {"single_qubit": int(N * p), "two_qubit": 0}
    pairs = set()
    for i in range(0, N - 1, 2):
        pairs.add((i, i + 1))
    for i in range(1, N - 1, 2):
        pairs.add((i, i + 1))
    pairs.add((N - 1, 0))
    two_per_step = len(pairs)
    return {"single_qubit": 0, "two_qubit": int(two_per_step * T * p)}
