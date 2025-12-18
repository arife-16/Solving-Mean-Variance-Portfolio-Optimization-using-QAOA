import time
import math
import random
import numpy as np
from typing import Dict, Any
from .data import generate_synthetic_returns, compute_mu_sigma
from .formulations import energies_full, energies_full_mad, energies_full_mvo_tc
from .classical import brute_force_k_hot, brute_force_from_energies
from .qaoa_core import dicke_state, warm_start_state, qaoa_expectation, qaoa_cvar, gate_counts, qaoa_expectation_shots, qaoa_cvar_shots, qaoa_expectation_ops, qaoa_cvar_ops, evolve_state, evolve_state_ops, compute_overlap
from .data import generate_transaction_costs
from .mip import solve_mvo_milp
from .adapt_qaoa import adapt_qaoa_pairs, build_pairs
from .adapt_qaoa import adapt_qaoa

class PortfolioPipeline:
    def __init__(self, seed: int = 1):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _get_problem(self, N: int, K: int, q: float) -> Dict[str, Any]:
        rets = generate_synthetic_returns(N, 60, self.seed)
        mu, sigma = compute_mu_sigma(rets)
        tc = generate_transaction_costs(N, self.seed)
        return {"N": N, "K": K, "q": q, "means": mu, "cov": sigma, "returns": rets, "tc": tc}

    def run_standard(self, N: int, K: int, q: float, p: int, mixer: str = "xy", T: int = 1, warm_start: bool = False, alpha: float = 0.2, samples: int = 32, refine_iters: int = 20, refine_step: float = 0.05, formulation: str = "mvo", lam_tc: float = 0.1, shots: int = 0, noise_p: float = 0.0, solver: str = "bruteforce") -> Dict[str, Any]:
        start = time.time()
        problem = self._get_problem(N=N, K=K, q=q)
        if formulation == "mvo":
            energies = energies_full(problem["means"], problem["cov"], q, N)
        elif formulation == "mad":
            energies = energies_full_mad(problem["returns"], q, N)
        elif formulation == "cvar":  
            from .formulations_extended import energies_full_cvar
            energies = energies_full_cvar(problem["returns"], q, N, alpha=0.05)
        else:
            energies = energies_full_mvo_tc(problem["means"], problem["cov"], q, N, problem["tc"], lam_tc)
        psi0 = warm_start_state(problem["means"], problem["cov"], K) if warm_start else dicke_state(N, K)
        def f(theta):
            if shots and shots > 0:
                return qaoa_expectation_shots(psi0, energies, N, K, theta, mixer=mixer, T=T, shots=shots, noise_p=noise_p)
            return qaoa_expectation(psi0, energies, N, K, theta, mixer=mixer, T=T)
        best_x = None
        best_y = math.inf
        for _ in range(samples):
            x = np.random.uniform(-2.0, 2.0, size=2 * p)
            y = f(x)
            if y < best_y:
                best_x, best_y = x, y
        for _ in range(refine_iters):
            cand = best_x + np.random.normal(0.0, refine_step, size=best_x.shape)
            y = f(cand)
            if y < best_y:
                best_x, best_y = cand, y
        if shots and shots > 0:
            cvar = qaoa_cvar_shots(psi0, energies, N, K, best_x, alpha, mixer=mixer, T=T, shots=shots, noise_p=noise_p)
        else:
            cvar = qaoa_cvar(psi0, energies, N, K, best_x, alpha, mixer=mixer, T=T)
        if solver == "milp" and formulation in ("mvo", "mvo_tc"):
            tc = problem["tc"] if formulation == "mvo_tc" else None
            sol = solve_mvo_milp(problem["means"], problem["cov"], q, N, K, tc, lam_tc)
            if sol is not None:
                emin, z_opt = sol
            else:
                emin, z_opt = brute_force_from_energies(energies, N, K)
        else:
            emin, z_opt = brute_force_from_energies(energies, N, K)
        psi = evolve_state(psi0, energies, N, best_x, mixer=mixer, T=T)
        overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots)
        gates = gate_counts(N, p, mixer, T)
        end = time.time()
        return {"best_energy": float(best_y), "optimal_energy": float(emin), "energy_gap": float(best_y - emin), "cvar": float(cvar), "overlap": float(overlap), "params": best_x.tolist(), "gate_counts": gates, "duration_sec": float(end - start), "solver_used": solver, "shots": int(shots), "noise_p": float(noise_p)}

    def run_adapt(self, N: int, K: int, q: float, max_layers: int, mixer: str = "xy", T: int = 1, warm_start: bool = False, alpha: float = 0.2, formulation: str = "mvo", lam_tc: float = 0.1, pool: str = "ring", shots: int = 0, noise_p: float = 0.0, pairs_mode: str = "ring") -> Dict[str, Any]:
        start = time.time()
        problem = self._get_problem(N=N, K=K, q=q)
        if formulation == "mvo":
            energies = energies_full(problem["means"], problem["cov"], q, N)
        elif formulation == "mad":
            energies = energies_full_mad(problem["returns"], q, N)
        elif formulation == "cvar":  
            from .formulations_extended import energies_full_cvar
            energies = energies_full_cvar(problem["returns"], q, N, alpha=0.05)
            
        else:
            energies = energies_full_mvo_tc(problem["means"], problem["cov"], q, N, problem["tc"], lam_tc)
        psi0 = warm_start_state(problem["means"], problem["cov"], K) if warm_start else dicke_state(N, K)
        if pool == "pairs":
            if mixer == "x":
                pairs = []
            else:
                pairs = build_pairs(N, pairs_mode)
            theta, ops, best, layers, gates = adapt_qaoa_pairs(psi0, energies, N, K, max_layers, pairs, T=T)
            if shots and shots > 0:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            else:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            emin, z_opt = brute_force_from_energies(energies, N, K)
            psi = evolve_state_ops(psi0, energies, N, theta, ops, T=T)
            overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots)
            end = time.time()
            return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - start), "shots": int(shots), "noise_p": float(noise_p), "pairs_mode": pairs_mode}
        else:
            theta, best, layers = adapt_qaoa(psi0, energies, N, K, max_layers, mixer=mixer, T=T)
            if shots and shots > 0:
                cvar = qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T, shots=shots, noise_p=noise_p)
            else:
                cvar = qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T)
        emin, z_opt = brute_force_from_energies(energies, N, K)
        gates = gate_counts(N, int(len(theta) // 2), mixer, T)
        psi = evolve_state(psi0, energies, N, theta, mixer=mixer, T=T)
        overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots)
        end = time.time()
        return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - start), "shots": int(shots), "noise_p": float(noise_p)}
