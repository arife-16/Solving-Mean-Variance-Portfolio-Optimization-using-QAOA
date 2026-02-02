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
                # Normalize orientation to (N_assets, T) and clean NaNs/Infs
                if isinstance(rets, np.ndarray):
                    R = rets.copy()
                else:
                    R = np.array(rets, dtype=float)
                if R.ndim != 2:
                    R = np.atleast_2d(R)
                # If first dim isn't N but second is, transpose
                if R.shape[0] != N and R.shape[1] == N:
                    R = R.T
                # Drop columns with any NaNs
                mask = ~np.isnan(R).any(axis=0)
                R = R[:, mask]
                # Replace remaining infs with finite values
                R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
                # Enforce asset dimension equals requested N (subset if larger)
                if R.shape[0] > N:
                    R = R[:N, :]
                elif R.shape[0] < N:
                    N = R.shape[0]
                rets = R
                mu, sigma = annualized_mu_sigma(rets)
            except Exception:
                rets = None
        if rets is None and prices_csv:
            prices = load_prices_csv(prices_csv)
            rets = returns_from_prices(prices)
            mu, sigma = compute_mu_sigma(rets)
        if rets is None:
            rets = generate_synthetic_returns(N, 60, self.seed)
            mu, sigma = compute_mu_sigma(rets)
        if tc_csv:
            tc = load_transaction_costs_csv(tc_csv)
        else:
            tc = generate_transaction_costs(N, self.seed)
        return {"N": N, "K": K, "q": q, "means": mu, "cov": sigma, "returns": rets, "tc": tc}

    def run_standard(self, N: int, K: int, q: float, p: int, mixer: str = "xy", T: int = 1, warm_start: bool = False, alpha: float = 0.2, samples: int = 32, refine_iters: int = 20, refine_step: float = 0.05, formulation: str = "mvo", lam_tc: float = 0.1, shots: int = 0, noise_p: float = 0.0, solver: str = "bruteforce", objective: str = "expectation", noise_model: str = "depolarizing", prices_csv: Optional[str] = None, tc_csv: Optional[str] = None, tickers: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None, penalty: float = 100.0) -> Dict[str, Any]:
        t0 = time.time()
        problem = self._get_problem(N=N, K=K, q=q, prices_csv=prices_csv, tc_csv=tc_csv, tickers=tickers, start=start, end=end)
        N = int(problem["N"])
        K = int(min(K, N))
        p = int(p)
        
        # Use subspace method for large N to avoid full state vector allocation
        if N > 16:
            from .subspace import generate_basis, compute_energies_subspace, qaoa_expectation_subspace, qaoa_cvar_subspace, evolve_state_subspace, compute_overlap_subspace
            
            states = generate_basis(N, K)
            energies = compute_energies_subspace(states, problem["means"], problem["cov"], q, N, penalty=penalty)
            psi0_sub = None
            if warm_start:
                try:
                    from .mip import relax_mvo_qp
                    x = relax_mvo_qp(problem["means"], problem["cov"], 1.0, N, K)
                except Exception:
                    x = None
                if x is None:
                    p_prob = 1 / (1 + np.exp(-problem["means"]))
                    w = np.array([np.log(p_prob[i] / (1 - p_prob[i])) for i in range(N)])
                else:
                    w = x
                amp = np.zeros(len(states), dtype=float)
                for idx, z in enumerate(states):
                    s = 0.0
                    for i in range(N):
                        if (int(z) >> i) & 1:
                            s += w[i]
                    amp[idx] = math.exp(s)
                if amp.sum() > 0:
                    psi0_sub = (amp / np.linalg.norm(amp)).astype(complex)
            
            def f(theta):
                if objective == "cvar":
                    return qaoa_cvar_subspace(states, energies, N, K, theta, alpha=alpha, mixer=mixer, T=T, psi0=psi0_sub)
                return qaoa_expectation_subspace(states, energies, N, K, theta, mixer=mixer, T=T, psi0=psi0_sub)
            try:
                from scipy.optimize import minimize
                x0 = np.zeros(2 * p)
                res = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
                best_x = res.x
                best_y = res.fun
            except Exception:
                best_x = np.zeros(2 * p)
                best_y = f(best_x)
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
            
            psi = evolve_state_subspace(states, energies, N, best_x, mixer=mixer, T=T, psi0=psi0_sub)
            overlap = compute_overlap_subspace(psi, states, z_opt)
            gates = gate_counts(N, p, mixer, T)
            end = time.time()
            
            return {
                "best_energy": float(best_y),
                "optimal_energy": float(emin),
                "energy_gap": float(best_y - emin),
                "cvar": float(qaoa_cvar_subspace(states, energies, N, K, best_x, alpha=alpha, mixer=mixer, T=T, psi0=psi0_sub)) if objective == "cvar" else 0.0,
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
        # Stabilize energies to avoid NaN/Inf propagation
        energies = np.nan_to_num(energies, nan=0.0, posinf=1e6, neginf=-1e6)
        if warm_start:
            psi0, thetas_ws = warm_start_state(problem["means"], problem["cov"], K)
            if mixer == "xy":
                mixer_to_use = "ws"
            else:
                mixer_to_use = mixer
        else:
            psi0 = dicke_state(N, K)
            thetas_ws = None
            mixer_to_use = mixer
        def f(theta):
            if objective == "cvar":
                if shots and shots > 0:
                    return qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer=mixer_to_use, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model, thetas=thetas_ws)
                return qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer=mixer_to_use, T=T, noise_model=noise_model, noise_p=noise_p, thetas=thetas_ws)
            else:
                if shots and shots > 0:
                    return qaoa_expectation_shots(psi0, energies, N, K, theta, mixer=mixer_to_use, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model, thetas=thetas_ws)
                return qaoa_expectation(psi0, energies, N, K, theta, mixer=mixer_to_use, T=T, noise_model=noise_model, noise_p=noise_p, thetas=thetas_ws)
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
            cvar = qaoa_cvar_shots(psi0, energies, N, K, best_x, alpha, mixer=mixer_to_use, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model, thetas=thetas_ws)
        else:
            cvar = qaoa_cvar(psi0, energies, N, K, best_x, alpha, mixer=mixer_to_use, T=T, noise_model=noise_model, noise_p=noise_p, thetas=thetas_ws)
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
        elif solver in ("lp",) and formulation == "mad":
            try:
                from .mip import solve_mad_lp
                sol = solve_mad_lp(problem["returns"], N, K)
            except Exception:
                sol = None
            if sol is not None:
                emin, z_opt = sol
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
        t0 = time.time()
        problem = self._get_problem(N=N, K=K, q=q, prices_csv=prices_csv, tc_csv=tc_csv, tickers=tickers, start=start, end=end)
        N = int(problem["N"])
        K = int(min(K, N))
        if N > 16:
            from .subspace import generate_basis, compute_energies_subspace, qaoa_cvar_subspace, qaoa_expectation_subspace, evolve_state_subspace, compute_overlap_subspace
            from .adapt_qaoa import adapt_qaoa_subspace
            states = generate_basis(N, K)
            energies = compute_energies_subspace(states, problem["means"], problem["cov"], q, N, penalty=penalty)
            psi0_sub = None
            if warm_start:
                try:
                    from .mip import relax_mvo_qp
                    x = relax_mvo_qp(problem["means"], problem["cov"], 1.0, N, K)
                except Exception:
                    x = None
                if x is None:
                    p_prob = 1 / (1 + np.exp(-problem["means"]))
                    w = np.array([np.log(p_prob[i] / (1 - p_prob[i])) for i in range(N)])
                else:
                    w = x
                amp = np.zeros(len(states), dtype=float)
                for idx, z in enumerate(states):
                    s = 0.0
                    for i in range(N):
                        if (int(z) >> i) & 1:
                            s += w[i]
                    amp[idx] = math.exp(s)
                if amp.sum() > 0:
                    psi0_sub = (amp / np.linalg.norm(amp)).astype(complex)
            theta, best, layers, trace = adapt_qaoa_subspace(states, energies, N, K, max_layers, mixer=mixer, T=T, objective=objective, alpha=alpha, psi0=psi0_sub)
            emin_idx = int(np.argmin(energies))
            emin = float(energies[emin_idx])
            z_opt = int(states[emin_idx])
            psi = evolve_state_subspace(states, energies, N, theta, mixer=mixer, T=T, psi0=psi0_sub)
            overlap = compute_overlap_subspace(psi, states, z_opt)
            gates = gate_counts(N, int(len(theta) // 2), mixer, T)
            end = time.time()
            cvar = qaoa_cvar_subspace(states, energies, N, K, theta, alpha=alpha, mixer=mixer, T=T, psi0=psi0_sub) if objective == "cvar" else 0.0
            return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - t0), "shots": int(shots), "noise_p": float(noise_p), "noise_model": noise_model, "objective": objective, "trace": trace}
        
        if formulation == "mvo":
            energies = energies_full(problem["means"], problem["cov"], q, N, K=K, penalty=penalty)
        elif formulation == "mad":
            energies = energies_full_mad(problem["returns"], q, N, K=K, penalty=penalty)
        elif formulation == "cvar":  
            from .formulations_extended import energies_full_cvar
            energies = energies_full_cvar(problem["returns"], q, N, alpha=0.05)
            
        else:
            energies = energies_full_mvo_tc(problem["means"], problem["cov"], q, N, problem["tc"], lam_tc, K=K, penalty=penalty)
        if warm_start:
            psi0, thetas_ws = warm_start_state(problem["means"], problem["cov"], K)
            mixer_to_use = "ws" if mixer == "xy" else mixer
        else:
            psi0 = dicke_state(N, K)
            thetas_ws = None
            mixer_to_use = mixer
        if pool == "pairs":
            if mixer == "x":
                pairs = []
            else:
                pairs = build_pairs(N, pairs_mode)
            try:
                from .adapt_qaoa import adapt_qaoa_pairs_gradient
                theta, ops, best, layers, gates, trace = adapt_qaoa_pairs_gradient(psi0, energies, N, K, max_layers, pairs, T=T, objective=objective, alpha=alpha)
            except Exception:
                theta, ops, best, layers, gates, trace = adapt_qaoa_pairs(psi0, energies, N, K, max_layers, pairs, T=T, objective=objective, alpha=alpha)
            if shots and shots > 0:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            else:
                cvar = qaoa_cvar_ops(psi0, energies, N, theta, alpha, ops, T=T)
            emin, z_opt = brute_force_from_energies(energies, N, K)
            psi = evolve_state_ops(psi0, energies, N, theta, ops, T=T)
            overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots, noise_model=noise_model)
            end = time.time()
            return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - t0), "shots": int(shots), "noise_p": float(noise_p), "pairs_mode": pairs_mode, "noise_model": noise_model, "objective": objective, "trace": trace}
        else:
            theta, best, layers, trace = adapt_qaoa(psi0, energies, N, K, max_layers, mixer=mixer_to_use, T=T, objective=objective, alpha=alpha)
            if shots and shots > 0:
                cvar = qaoa_cvar_shots(psi0, energies, N, K, theta, alpha, mixer=mixer_to_use, T=T, shots=shots, noise_p=noise_p, noise_model=noise_model, thetas=thetas_ws)
            else:
                cvar = qaoa_cvar(psi0, energies, N, K, theta, alpha, mixer=mixer_to_use, T=T, noise_model=noise_model, noise_p=noise_p, thetas=thetas_ws)
        emin, z_opt = brute_force_from_energies(energies, N, K)
        gates = gate_counts(N, int(len(theta) // 2), mixer_to_use, T)
        psi = evolve_state(psi0, energies, N, theta, mixer=mixer_to_use, T=T, thetas=thetas_ws)
        overlap = compute_overlap(psi, z_opt, noise_p=noise_p, shots=shots, noise_model=noise_model)
        end = time.time()
        return {"best_energy": float(best), "optimal_energy": float(emin), "energy_gap": float(best - emin), "cvar": float(cvar), "overlap": float(overlap), "params": theta.tolist(), "layers": int(layers), "gate_counts": gates, "duration_sec": float(end - t0), "shots": int(shots), "noise_p": float(noise_p), "noise_model": noise_model, "objective": objective, "trace": trace}