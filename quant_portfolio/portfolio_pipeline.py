import time
import math
import random
import numpy as np
from typing import Dict, Any, Optional, List
from .data import generate_synthetic_returns, compute_mu_sigma, load_prices_csv, returns_from_prices, load_transaction_costs_csv
from .formulations import energies_full, energies_full_mad, energies_full_mvo_tc
from .classical import brute_force_k_hot, brute_force_from_energies
from .qaoa_core import dicke_state, warm_start_state, qaoa_expectation, qaoa_cvar, gate_counts, qaoa_expectation_shots, qaoa_cvar_shots, qaoa_expectation_ops, qaoa_cvar_ops, evolve_state, evolve_state_ops, compute_overlap
from .data import generate_transaction_costs
from .mip import solve_mvo_milp, solve_mvo_miqp
from .adapt_qaoa import adapt_qaoa_pairs, build_pairs
from .adapt_qaoa import adapt_qaoa

class PortfolioPipeline:
    def __init__(self, seed: int = 1):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def _get_problem(self, N: int, K: int, q: float, prices_csv: Optional[str] = None, tc_csv: Optional[str] = None, tickers: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, Any]:
        rets = None
        if tickers and start and end:
            try:
                from .data_loader import fetch_real_data, compute_returns_from_prices, annualized_mu_sigma
                prices_df = fetch_real_data(tickers, start, end)
                rets = compute_returns_from_prices(prices_df, method="log")
                mu, sigma, rets = annualized_mu_sigma(rets)  # ← Get transposed returns
            except Exception:
                rets = None
        if rets is None and prices_csv:
            prices = load_prices_csv(prices_csv)
            rets = returns_from_prices(prices)
            mu, sigma = compute_mu_sigma(rets)
        if rets is None:
            rets = generate_synthetic_returns(N, 60, self.seed)
            rets = rets.T
            mu, sigma = compute_mu_sigma(rets)
        if tc_csv:
            tc = load_transaction_costs_csv(tc_csv)
        else:
            tc = generate_transaction_costs(N, self.seed)
        return {"N": N, "K": K, "q": q, "means": mu, "cov": sigma, "returns": rets, "tc": tc}

    def run_standard(self, N: int, K: int, q: float, p: int, mixer: str = "xy", T: int = 1, warm_start: bool = False, alpha: float = 0.2, samples: int = 32, refine_iters: int = 20, refine_step: float = 0.05, formulation: str = "mvo", lam_tc: float = 0.1, shots: int = 0, noise_p: float = 0.0, solver: str = "bruteforce", objective: str = "expectation", noise_model: str = "depolarizing", prices_csv: Optional[str] = None, tc_csv: Optional[str] = None, tickers: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None, penalty: float = 100.0) -> Dict[str, Any]:
        t0 = time.time()
        problem = self._get_problem(N=N, K=K, q=q, prices_csv=prices_csv, tc_csv=tc_csv, tickers=tickers, start=start, end=end)
        
        # Use subspace method for large N to avoid full state vector allocation
        if N > 24:
            from .subspace import generate_basis, compute_energies_subspace, qaoa_expectation_subspace, evolve_state_subspace, compute_overlap_subspace
            
            states = generate_basis(N, K)
            # Default to MVO formulation for subspace
            energies = compute_energies_subspace(states, problem["means"], problem["cov"], q, N, penalty=penalty)
            
            def f(theta):
                return qaoa_expectation_subspace(states, energies, N, K, theta, mixer=mixer, T=T)
            
            best_x = None
            best_y = math.inf
            
            # Simple random search + local refinement
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
            
            # Find optimal classically using subspace energies
            min_idx = np.argmin(energies)
            emin = float(energies[min_idx])
            z_opt = int(states[min_idx])
            
            psi = evolve_state_subspace(states, energies, N, best_x, mixer=mixer, T=T)
            overlap = compute_overlap_subspace(psi, states, z_opt)
            gates = gate_counts(N, p, mixer, T)
            end = time.time()
            
            return {
                "best_energy": float(best_y),
                "optimal_energy": float(emin),
                "energy_gap": float(best_y - emin),
                "cvar": 0.0, # Not implemented for subspace yet
                "overlap": float(overlap),
                "params": best_x.tolist(),
                "gate_counts": gates,
                "duration_sec": float(end - t0),
                "solver_used": "subspace_brute",
                "shots": 0,
                "noise_p": 0.0,
                "noise_model": "none",
                "objective": objective
            }

        if formulation == "mvo":
            energies = energies_full(problem["means"], problem["cov"], q, N, K=K, penalty=penalty)
        elif formulation == "mad":
            energies = energies_full_mad(problem["returns"], q, N, K=K, penalty=penalty)
        elif formulation == "cvar":  
            from .formulations_extended import energies_full_cvar
            energies = energies_full_cvar(problem["returns"], q, N, alpha=0.05)
        else:
            energies = energies_full_mvo_tc(problem["means"], problem["cov"], q, N, problem["tc"], lam_tc, K=K, penalty=penalty)
        psi0 = warm_start_state(problem["means"], problem["cov"], K) if warm_start else dicke_state(N, K)
        def f(theta):
            if objective == "cvar":
                if shots and shots > 0:
                    return qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model)
                return qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T, noise_model=noise_model, noise_p=noise_p)
            else:
                if shots and shots > 0:
                    return qaoa_expectation_shots(psi0, energies, N, K, theta, mixer=mixer, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model)
                return qaoa_expectation(psi0, energies, N, K, theta, mixer=mixer, T=T, noise_model=noise_model, noise_p=noise_p)
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
            cvar = qaoa_cvar_shots(psi0, energies, N, K, best_x, alpha, mixer=mixer, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model)
        else:
            cvar = qaoa_cvar(psi0, energies, N, K, best_x, alpha, mixer=mixer, T=T, noise_model=noise_model, noise_p=noise_p)
        if solver in ("milp", "miqp") and formulation in ("mvo", "mvo_tc"):
            tc = problem["tc"] if formulation == "mvo_tc" else None
            if solver == "milp":
                sol = solve_mvo_milp(problem["means"], problem["cov"], q, N, K, tc, lam_tc)
            else:
                sol = solve_mvo_miqp(problem["means"], problem["cov"], q, N, K, tc, lam_tc)
            if sol is not None:
                emin, z_opt = sol
            else:
                try:
                    from .mip import relax_mvo_qp
                    x = relax_mvo_qp(problem["means"], problem["cov"], q, N, K)
                except Exception:
                    x = None
                if x is not None:
                    idx = np.argsort(-x)[:K]
                    z = 0
                    for i in idx:
                        z |= (1 << int(i))
                    emin = float(energies[z])
                    z_opt = int(z)
                else:
                    emin, z_opt = brute_force_from_energies(energies, N, K)
        else:
            emin, z_opt = brute_force_from_energies(energies, N, K)
        psi = evolve_state(psi0, energies, N, best_x, mixer=mixer, T=T)
        overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots, noise_model=noise_model)
        gates = gate_counts(N, p, mixer, T)
        end = time.time()
        return {"best_energy": float(best_y), "optimal_energy": float(emin), "energy_gap": float(best_y - emin), "cvar": float(cvar), "overlap": float(overlap), "params": best_x.tolist(), "gate_counts": gates, "duration_sec": float(end - t0), "solver_used": solver, "shots": int(shots), "noise_p": float(noise_p), "noise_model": noise_model, "objective": objective}

    def run_adapt(self, N: int, K: int, q: float, max_layers: int, mixer: str = "xy", T: int = 1, warm_start: bool = False, alpha: float = 0.2, formulation: str = "mvo", lam_tc: float = 0.1, pool: str = "ring", shots: int = 0, noise_p: float = 0.0, pairs_mode: str = "ring", objective: str = "expectation", noise_model: str = "depolarizing", prices_csv: Optional[str] = None, tc_csv: Optional[str] = None, tickers: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None, penalty: float = 100.0) -> Dict[str, Any]:
        if N > 24:
            raise NotImplementedError("ADAPT-QAOA is not yet supported for N > 24 (requires subspace implementation). Please use run_standard for large N.")
        
        t0 = time.time()
        problem = self._get_problem(N=N, K=K, q=q, prices_csv=prices_csv, tc_csv=tc_csv, tickers=tickers, start=start, end=end)
        if formulation == "mvo":
            energies = energies_full(problem["means"], problem["cov"], q, N, K=K, penalty=penalty)
        elif formulation == "mad":
            energies = energies_full_mad(problem["returns"], q, N, K=K, penalty=penalty)
        elif formulation == "cvar":  
            from .formulations_extended import energies_full_cvar
            energies = energies_full_cvar(problem["returns"], q, N, alpha=0.05)
            
        else:
            energies = energies_full_mvo_tc(problem["means"], problem["cov"], q, N, problem["tc"], lam_tc, K=K, penalty=penalty)
        psi0 = warm_start_state(problem["means"], problem["cov"], K) if warm_start else dicke_state(N, K)
        if pool == "pairs":
            if mixer == "x":
                pairs = []
            else:
                pairs = build_pairs(N, pairs_mode)
            theta, ops, best, layers, gates = adapt_qaoa_pairs(psi0, energies, N, K, max_layers, pairs, T=T, objective=objective, alpha=alpha)
            if shots and shots > 0:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            else:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            emin, z_opt = brute_force_from_energies(energies, N, K)
            psi = evolve_state_ops(psi0, energies, N, theta, ops, T=T)
            overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots, noise_model=noise_model)
            end = time.time()
            return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - t0), "shots": int(shots), "noise_p": float(noise_p), "pairs_mode": pairs_mode, "noise_model": noise_model, "objective": objective}
        else:
            theta, best, layers = adapt_qaoa(psi0, energies, N, K, max_layers, mixer=mixer, T=T, objective=objective, alpha=alpha)
            if shots and shots > 0:
                cvar = qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model)
            else:
                cvar = qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer=mixer, T=T, noise_model=noise_model, noise_p=noise_p)
        emin, z_opt = brute_force_from_energies(energies, N, K)
        gates = gate_counts(N, int(len(theta) // 2), mixer, T)
        psi = evolve_state(psi0, energies, N, theta, mixer=mixer, T=T)
        overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots, noise_model=noise_model)
        end = time.time()
        return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - t0), "shots": int(shots), "noise_p": float(noise_p), "noise_model": noise_model, "objective": objective}
