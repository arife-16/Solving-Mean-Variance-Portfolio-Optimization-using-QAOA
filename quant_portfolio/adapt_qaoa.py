import numpy as np
from .qaoa_core import qaoa_expectation, qaoa_expectation_ops

def adapt_qaoa(psi0, energies, N, K, max_layers, mixer="xy", T=1, samples=8, step=0.1):
    theta = np.zeros(2)
    best = qaoa_expectation(psi0, energies, N, K, theta, mixer, T)
    layers = 0
    while layers < max_layers:
        cand_best_val = float("inf")
        cand_best_theta = None
        for _ in range(samples):
            t = np.concatenate([theta, np.random.uniform(-1.0, 1.0, size=2)])
            v = qaoa_expectation(psi0, energies, N, K, t, mixer, T)
            if v < cand_best_val:
                cand_best_val = v
                cand_best_theta = t
        if cand_best_val < best - 1e-9:
            theta = cand_best_theta
            best = cand_best_val
            layers += 1
        else:
            break
    return theta, best, layers

def adapt_qaoa_pairs(psi0, energies, N, K, max_layers, pairs, T=1, samples=8):
    theta = np.zeros(0)
    ops = []
    gate_two = 0
    best = float("inf")
    layers = 0
    while layers < max_layers:
        cand_best_val = float("inf")
        cand_theta = None
        cand_ops = None
        cand_gate_two = None
        for (i, j) in pairs:
            for _ in range(samples):
                t = np.concatenate([theta, np.random.uniform(-1.0, 1.0, size=2)])
                o = ops + [("xy_pair", (i, j))]
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
        else:
            break
    return theta, ops, best, layers, {"single_qubit": 0, "two_qubit": int(gate_two)}

def build_pairs(N: int, mode: str):
    if mode == "all":
        res = []
        for i in range(N):
            for j in range(i + 1, N):
                res.append((i, j))
        return res
    res = [(i, i + 1) for i in range(N - 1)] + [(N - 1, 0)]
    return res
