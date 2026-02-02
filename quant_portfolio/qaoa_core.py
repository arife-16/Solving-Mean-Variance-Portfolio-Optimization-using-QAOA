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
    try:
        from .mip import relax_mvo_qp
        x = relax_mvo_qp(mu, sigma, 1.0, N, K)
    except Exception:
        x = None
    if x is None:
        p = 1 / (1 + np.exp(-mu))
        w = np.array([np.log(p[i] / (1 - p[i])) for i in range(N)])
    else:
        w = x
    c = np.clip(w, 0.0, 1.0)
    thetas = 2.0 * np.arcsin(np.sqrt(c))
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
        return dicke_state(N, K), thetas
    psi = amp / np.linalg.norm(amp)
    return psi.astype(complex), thetas

def rx(theta):
    c = np.cos(theta / 2.0)
    s = -1j * np.sin(theta / 2.0)
    return np.array([[c, s], [s, c]], dtype=complex)

def ry(theta):
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)

def rz(theta):
    return np.array([[np.exp(-1j * theta / 2.0), 0], [0, np.exp(1j * theta / 2.0)]], dtype=complex)

def xy_unitary(beta):
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    H = np.kron(X, X) + np.kron(Y, Y)
    vals, vecs = np.linalg.eigh(H)
    U = (vecs @ np.diag(np.exp(-1j * beta * vals)) @ vecs.conj().T)
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
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
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    U = np.asarray(U, dtype=np.complex128, order='C')
    s = np.asarray(s, dtype=np.complex128, order='C')
    s = np.einsum('ab,bc->ac', U, s, optimize=True)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    psi = s.reshape([2, 2] + [2] * (N - 2))
    psi = np.moveaxis(psi, [0, 1], [i, j])
    out = psi.reshape(1 << N)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def apply_rx_all(state, beta, N):
    psi = state.reshape([2] * N)
    U = rx(2 * beta)
    for i in range(N):
        psi = np.tensordot(U, psi, axes=[[1], [i]])
    psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
    out = psi.reshape(1 << N)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def apply_warm_start_mixer(state, beta, thetas, N):
    """
    Implements the Modified Mixer from Egger et al. (2021), Eq. 2.
    Unitary: Prod_i [ Ry(theta_i) Rz(-2*beta) Ry(-theta_i) ]
    
    CRITICAL: This rotation sequence preserves the initial state |phi_0>
    as the ground state of the mixer Hamiltonian H_M^(ws).
    """
    psi = state.reshape([2] * N)
    for i in range(N):
        U1 = ry(-thetas[i])
        psi = np.tensordot(U1, psi, axes=[[1], [i]])
        U2 = rz(-2.0 * beta)
        psi = np.tensordot(U2, psi, axes=[[1], [i]])
        U3 = ry(thetas[i])
        psi = np.tensordot(U3, psi, axes=[[1], [i]])
    psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
    out = psi.reshape(1 << N)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out

def apply_xy_ring(state, beta, N, T):
    U = xy_unitary(4 * beta / T)
    psi = state
    for _ in range(T):
        for i in range(0, N - 1, 2):
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        for i in range(1, N - 1, 2):
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        psi = apply_two_qubit(psi, U, N - 1, 0, N)
    psi = np.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(psi)
    if norm > 1e-12:
        psi = psi / norm
    return psi

def apply_xy_qampa(state, beta, N, T, K):
    psi = state
    s = K / max(N, 1)
    for _ in range(T):
        for i in range(0, N - 1, 2):
            w = beta * (1.0 + 0.5 * np.cos(2.0 * np.pi * i / max(N, 1))) * s
            U = xy_unitary(4 * w / 1)
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        for i in range(1, N - 1, 2):
            w = beta * (1.0 + 0.5 * np.cos(2.0 * np.pi * i / max(N, 1))) * s
            U = xy_unitary(4 * w / 1)
            psi = apply_two_qubit(psi, U, i, i + 1, N)
        w = beta * (1.0 + 0.5 * np.cos(2.0 * np.pi * (N - 1) / max(N, 1))) * s
        U = xy_unitary(4 * w / 1)
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

def qaoa_layer(state, energies, N, beta, mixer, T, thetas=None):
    if mixer == "x":
        return apply_rx_all(state, beta, N)
    if mixer == "ws" and thetas is not None:
        return apply_warm_start_mixer(state, beta, thetas, N)
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

def apply_bitflip(probs, p, N):
    if p <= 0.0:
        return probs
    res = probs.copy()
    dim = probs.shape[0]
    for i in range(N):
        flip = np.zeros(dim, dtype=float)
        for z in range(dim):
            flip[z ^ (1 << i)] = probs[z]
        res = (1.0 - p) * res + p * flip
    return res

def apply_phaseflip(probs, p, N):
    return probs

def apply_noise(probs, model, p, N):
    if p <= 0.0:
        return probs
    if model == "depolarizing":
        return apply_depolarizing(probs, p)
    if model == "bitflip":
        return apply_bitflip(probs, p, N)
    if model == "phaseflip":
        return apply_phaseflip(probs, p, N)
    return probs

def qaoa_expectation_shots(psi0, energies, N, K, theta, mixer="xy", T=1, shots=1024, noise_p=0.0, noise_model="depolarizing", thetas=None):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        if mixer == "qampa":
            psi = apply_xy_qampa(psi, b[t], N, T, K)
        else:
            psi = qaoa_layer(psi, energies, N, b[t], mixer, T, thetas)
    probs = np.real(psi.conj() * psi)
    probs = probs / max(probs.sum(), 1e-12)
    dim = probs.shape[0]
    est = 0.0
    for _ in range(max(shots, 1)):
        z = np.random.choice(np.arange(dim), p=probs)
        if noise_model == "bitflip" and noise_p > 0.0:
            for i in range(N):
                if np.random.rand() < noise_p:
                    z ^= (1 << i)
        est += energies[z]
    est = est / max(shots, 1)
    return est

def qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer="xy", T=1, shots=1024, noise_p=0.0, noise_model="depolarizing", thetas=None):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        if mixer == "qampa":
            psi = apply_xy_qampa(psi, b[t], N, T, K)
        else:
            psi = qaoa_layer(psi, energies, N, b[t], mixer, T, thetas)
    probs = np.real(psi.conj() * psi)
    probs = probs / max(probs.sum(), 1e-12)
    dim = probs.shape[0]
    samples = []
    for _ in range(max(shots, 1)):
        z = np.random.choice(np.arange(dim), p=probs)
        if noise_model == "bitflip" and noise_p > 0.0:
            for i in range(N):
                if np.random.rand() < noise_p:
                    z ^= (1 << i)
        samples.append(energies[z])
    order = np.argsort(samples)[::-1]
    thr = int(alpha * max(shots, 1))
    s = 0.0
    for k in range(min(thr, len(order))):
        s += samples[order[k]]
    return float(s / max(thr, 1))

def evolve_state(psi0, energies, N, theta, mixer="xy", T=1, thetas=None):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        psi = qaoa_layer(psi, energies, N, b[t], mixer, T, thetas)
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

def compute_overlap(psi, z_opt, noise_p=0.0, shots=0, noise_model="depolarizing"):
    probs = np.real(psi.conj() * psi)
    probs = apply_noise(probs, noise_model, noise_p, probs.shape[0].bit_length() - 1)
    probs = probs / max(probs.sum(), 1e-12)
    if shots and shots > 0:
        counts = np.random.multinomial(shots, probs)
        return float(counts[z_opt] / max(shots, 1))
    return float(probs[z_opt])

def qaoa_expectation(psi0, energies, N, K, theta, mixer="xy", T=1, noise_model="depolarizing", noise_p=0.0, thetas=None):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        if mixer == "qampa":
            psi = apply_xy_qampa(psi, b[t], N, T, K)
        else:
            psi = qaoa_layer(psi, energies, N, b[t], mixer, T, thetas)
    probs = np.real(psi.conj() * psi)
    probs = apply_noise(probs, noise_model, noise_p, N)
    probs = probs / max(probs.sum(), 1e-12)
    return float((probs * energies).sum())

def qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer="xy", T=1, noise_model="depolarizing", noise_p=0.0, thetas=None):
    g, b = np.split(theta, 2)
    psi = psi0.copy()
    for t in range(len(g)):
        phase = np.exp(-1j * g[t] * energies)
        psi = psi * phase
        if mixer == "qampa":
            psi = apply_xy_qampa(psi, b[t], N, T, K)
        else:
            psi = qaoa_layer(psi, energies, N, b[t], mixer, T, thetas)
    probs = np.real(psi.conj() * psi)
    probs = apply_noise(probs, noise_model, noise_p, N)
    probs = probs / max(probs.sum(), 1e-12)
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
