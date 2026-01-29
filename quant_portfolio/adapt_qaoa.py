import numpy as np
from .qaoa_core import qaoa_expectation, qaoa_expectation_ops, qaoa_cvar, qaoa_cvar_ops
from .subspace import qaoa_expectation_subspace, qaoa_cvar_subspace
import numpy as np
import numpy.linalg as LA
from .qaoa_core import evolve_state_ops

def select_operator_via_gradient(psi, pool, H_c):
    """
    Implements the Zhu et al. (2022) gradient criterion:
    Metric = | <psi | [H_C, A_k] | psi > |
    Avoids full VQE loop for operator selection.
    """
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Gxy = np.kron(X, X) + np.kron(Y, Y)
    def apply_generator(psi_vec, G, i, j, N):
        if i > j:
            i, j = j, i
        s = psi_vec.reshape([2] * N)
        s = np.moveaxis(s, [i, j], [0, 1])
        m = s.reshape(4, -1)
        m2 = G @ m
        s2 = m2.reshape([2, 2] + [2] * (N - 2))
        s2 = np.moveaxis(s2, [0, 1], [i, j])
        return s2.reshape(1 << N)
    best = None
    best_score = -np.inf
    N = int(np.log2(psi.shape[0]))
    for (i, j) in pool:
        psi_prime = apply_generator(psi, Gxy, i, j, N)
        term1 = np.vdot(psi, H_c * psi_prime)
        tmp = H_c * psi
        term2_vec = apply_generator(tmp, Gxy, i, j, N)
        term2 = np.vdot(psi, term2_vec)
        val = term1 - term2
        score = np.abs(val)
        if score > best_score:
            best_score = score
            best = (i, j)
    return best, float(best_score)
def adapt_qaoa(psi0, energies, N, K, max_layers, mixer="xy", T=1, samples=8, step=0.1, objective="expectation", alpha=0.2):
    theta = np.zeros(2)
    trace = []
    if objective == "cvar":
        best = qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer, T)
    else:
        best = qaoa_expectation(psi0, energies, N, K, theta, mixer, T)
    layers = 0
    trace.append(float(best))
    while layers < max_layers:
        cand_best_val = float("inf")
        cand_best_theta = None
        for _ in range(samples):
            t = np.concatenate([theta, np.random.uniform(-1.0, 1.0, size=2)])
            if objective == "cvar":
                v = qaoa_cvar(psi0, energies, N, K, t, alpha, mixer, T)
            else:
                v = qaoa_expectation(psi0, energies, N, K, t, mixer, T)
            if v < cand_best_val:
                cand_best_val = v
                cand_best_theta = t
        if cand_best_val < best - 1e-9:
            theta = cand_best_theta
            best = cand_best_val
            layers += 1
            trace.append(float(best))
        else:
            break
    return theta, best, layers, trace

def adapt_qaoa_subspace(dim_states, energies, N, K, max_layers, mixer="xy", T=1, samples=8, objective="expectation", alpha=0.2, psi0=None):
    theta = np.zeros(2)
    trace = []
    if objective == "cvar":
        best = qaoa_cvar_subspace(dim_states, energies, N, K, theta, alpha=alpha, mixer=mixer, T=T, psi0=psi0)
    else:
        best = qaoa_expectation_subspace(dim_states, energies, N, K, theta, mixer=mixer, T=T, psi0=psi0)
    layers = 0
    trace.append(float(best))
    while layers < max_layers:
        cand_best_val = float("inf")
        cand_best_theta = None
        for _ in range(samples):
            t = np.concatenate([theta, np.random.uniform(-1.0, 1.0, size=2)])
            if objective == "cvar":
                v = qaoa_cvar_subspace(dim_states, energies, N, K, t, alpha=alpha, mixer=mixer, T=T, psi0=psi0)
            else:
                v = qaoa_expectation_subspace(dim_states, energies, N, K, t, mixer=mixer, T=T, psi0=psi0)
            if v < cand_best_val:
                cand_best_val = v
                cand_best_theta = t
        if cand_best_val < best - 1e-9:
            theta = cand_best_theta
            best = cand_best_val
            layers += 1
            trace.append(float(best))
        else:
            break
    return theta, best, layers, trace

def adapt_qaoa_pairs(psi0, energies, N, K, max_layers, pairs, T=1, samples=8, objective="expectation", alpha=0.2):
    theta = np.zeros(0)
    ops = []
    gate_two = 0
    best = float("inf")
    layers = 0
    trace = []
    while layers < max_layers:
        cand_best_val = float("inf")
        cand_theta = None
        cand_ops = None
        cand_gate_two = None
        for (i, j) in pairs:
            for _ in range(samples):
                t = np.concatenate([theta, np.random.uniform(-1.0, 1.0, size=2)])
                o = ops + [("xy_pair", (i, j))]
                if objective == "cvar":
                    v = qaoa_cvar_ops(psi0, energies, N, t, alpha, o, T)
                else:
                    v = qaoa_expectation_ops(psi0, energies, N, t, o, T)
                if v < cand_best_val:
                    cand_best_val = v
                    cand_theta = t
                    cand_ops = o
                    cand_gate_two = gate_two + T
        if cand_best_val < best - 1e-9:
            theta = cand_theta
            ops = cand_ops
            gate_two = cand_gate_two
            best = cand_best_val
            layers += 1
            trace.append(float(best))
        else:
            break
    return theta, ops, best, layers, {"single_qubit": 0, "two_qubit": int(gate_two)}, trace

def adapt_qaoa_pairs_gradient(psi0, energies, N, K, max_layers, pairs, T=1, objective="expectation", alpha=0.2, gamma0=0.01, beta0=0.05):
    theta = np.zeros(0)
    ops = []
    gate_two = 0
    best = float("inf")
    layers = 0
    trace = []
    def eval_obj(t, olist):
        if objective == "cvar":
            return qaoa_cvar_ops(psi0, energies, N, t, alpha, olist, T)
        return qaoa_expectation_ops(psi0, energies, N, t, olist, T)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Gxy = np.kron(X, X) + np.kron(Y, Y)
    def apply_generator(psi, G, i, j, N):
        if i > j:
            i, j = j, i
        s = psi.reshape([2] * N)
        s = np.moveaxis(s, [i, j], [0, 1])
        m = s.reshape(4, -1)
        m2 = G @ m
        s2 = m2.reshape([2, 2] + [2] * (N - 2))
        s2 = np.moveaxis(s2, [0, 1], [i, j])
        return s2.reshape(1 << N)
    def compute_commutator_gradient(psi, energies, i, j):
        psi_prime = apply_generator(psi, Gxy, i, j, N)
        term1 = np.vdot(psi, energies * psi_prime)
        tmp = energies * psi
        term2_vec = apply_generator(tmp, Gxy, i, j, N)
        term2 = np.vdot(psi, term2_vec)
        val = term1 - term2
        return float(np.abs(val))
    while layers < max_layers:
        # t0 = np.concatenate([theta, np.array([gamma0, 0.0])])
        # v0 = eval_obj(t0, ops)
        best_score = float("inf")
        best_cand = None
        # Compute current evolved state under existing ops and theta
        psi_cur = evolve_state_ops(psi0, energies, N, theta if theta.size > 0 else np.zeros(0), ops, T)
        for (i, j) in pairs:
            score = -compute_commutator_gradient(psi_cur, energies, i, j)
            if score < best_score:
                best_score = score
                best_cand = (i, j)
        if best_cand is None:
            break
        ops = ops + [("xy_pair", best_cand)]
        gate_two += T
        theta = np.concatenate([theta, np.array([gamma0, 0.0])])
        try:
            from scipy.optimize import minimize
            res = minimize(lambda t: eval_obj(t, ops), theta, method="COBYLA", options={"maxiter": 200, "tol": 1e-4})
            theta = res.x
            val = res.fun
        except Exception:
            val = eval_obj(theta, ops)
            for _ in range(32):
                cand = theta + np.random.normal(0.0, 0.05, size=theta.shape)
                v = eval_obj(cand, ops)
                if v < val:
                    theta, val = cand, v
        if val < best - 1e-9:
            best = val
            layers += 1
            trace.append(float(best))
        else:
            break
    return theta, ops, best, layers, {"single_qubit": 0, "two_qubit": int(gate_two)}, trace

def build_pairs(N: int, mode: str):
    if mode == "all":
        res = []
        for i in range(N):
            for j in range(i + 1, N):
                res.append((i, j))
        return res
    res = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)]
    return res
