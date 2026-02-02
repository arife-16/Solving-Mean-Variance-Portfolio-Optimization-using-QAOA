
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
from quant_portfolio.portfolio_pipeline import PortfolioPipeline
from quant_portfolio.qaoa_core import warm_start_state, qaoa_expectation, evolve_state, compute_overlap, dicke_state
from quant_portfolio.mip import relax_mvo_qp
from quant_portfolio.adapt_qaoa import adapt_qaoa_pairs_gradient, build_pairs, qaoa_cvar_ops, evolve_state_ops
import time

def relax_mvo_scipy(mu, sigma, q, N, K):
    """
    Solves the continuous relaxation using scipy.optimize.minimize (SLSQP).
    Minimize q * x^T Sigma x - mu^T x
    Subject to sum(x) = K, 0 <= x <= 1
    """
    def obj(x):
        return q * (x @ sigma @ x) - mu @ x
    
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - K}
    ]
    bounds = [(0.0, 1.0) for _ in range(N)]
    x0 = np.full(N, K/N)
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints, tol=1e-6)
        if res.success:
            return res.x
    except Exception as e:
        print(f"Scipy Relaxation failed: {e}")
    return None

def solve_qp_sharpened(mu, sigma, q, N, K, alpha=2.0):
    """
    Solves the relaxation and then sharpens the result.
    w_new = sigmoid(alpha * (w - 0.5)) normalized
    """
    # Try CVXPY first, then Scipy
    x = None
    try:
        x = relax_mvo_qp(mu, sigma, q, N, K)
    except:
        pass
    
    if x is None:
        x = relax_mvo_scipy(mu, sigma, q, N, K)
        
    if x is None:
        return None
        
    # Sharpening heuristic
    w = np.array(x)
    # Clip to avoid numerical issues
    w = np.clip(w, 0.0, 1.0)
    # Simple power sharpening
    w_sharp = np.power(w, alpha)
    # Normalize to sum to K
    if w_sharp.sum() > 0:
        w_sharp = w_sharp / np.sum(w_sharp) * K
    w_sharp = np.clip(w_sharp, 0.0, 1.0)
    return w_sharp


def get_manual_ws_state(w, N, K):
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

def optimize_angles(psi0, energies, N, K, mixer, p, thetas_ws=None):
    def f(params):
        return qaoa_expectation(psi0, energies, N, K, params, mixer=mixer, T=1, thetas=thetas_ws)
    
    best_res = None
    best_val = float('inf')
    
    # Try a few random starts
    for _ in range(5):
        x0 = np.random.uniform(-0.5, 0.5, size=2*p)
        # For WS mixer, beta usually small, gamma small
        res = minimize(f, x0, method='COBYLA', options={'maxiter': 200})
        if res.fun < best_val:
            best_val = res.fun
            best_res = res
            
    return best_res.x, best_val

def run_experiment():
    pipeline = PortfolioPipeline()
    # N=12 Real Data
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "JPM", "V", "UNH", "WMT", "PG", "XOM"]
    start_date = "2018-01-01"
    end_date = "2020-01-01"
    
    print("Loading Data...")
    problem = pipeline._get_problem(N=12, K=6, q=1.0, tickers=tickers, start=start_date, end=end_date)
    N = problem["N"]
    K = problem["K"]
    means = problem["means"]
    cov = problem["cov"]
    
    from quant_portfolio.formulations import energies_full
    energies = energies_full(means, cov, 1.0, N, K=K)
    min_idx = np.argmin(energies)
    z_opt = min_idx # Index is the state
    emin = energies[min_idx]
    
    print(f"Optimal Energy: {emin}")
    
    results = []
    
    # 1. Standard Relaxation
    print("\n--- Standard Relaxation ---")
    try:
        import cvxpy
        print(f"CVXPY version: {cvxpy.__version__}")
    except ImportError:
        print("CVXPY not installed")

    w_std = relax_mvo_qp(means, cov, 1.0, N, K)
    
    if w_std is None:
        print("Relaxation failed (returned None). Trying Scipy...")
        w_std = relax_mvo_scipy(means, cov, 1.0, N, K)
        
    if w_std is None:
        print("Scipy Relaxation also failed. Using Logistic Fallback.")
        p_prob = 1 / (1 + np.exp(-means))
        w_std = p_prob
    
    # Normalize w_std to sum to K just to be safe for overlap calc
    w_std = w_std * (K / w_std.sum())
    
    psi_std, thetas_std = get_manual_ws_state(w_std, N, K)
    ov_std = compute_overlap(psi_std, z_opt)
    print(f"Initial Overlap: {ov_std:.4f}")
    
    # 2. Sharpened Relaxation
    print("\n--- Sharpened Relaxation (alpha=3) ---")
    w_shp = solve_qp_sharpened(means, cov, 1.0, N, K, alpha=3.0)
    if w_shp is None:
         print("Sharpened Relaxation failed. Using sharpened fallback.")
         w_shp = np.power(w_std, 3.0)
         if w_shp.sum() > 0:
             w_shp = w_shp / w_shp.sum() * K
         
    psi_shp, thetas_shp = get_manual_ws_state(w_shp, N, K)
    ov_shp = compute_overlap(psi_shp, z_opt)
    print(f"Initial Overlap: {ov_shp:.4f}")
    
    modes = [
        ("Std_Relax_WS_Mixer", psi_std, thetas_std, "ws"),
        ("Shp_Relax_WS_Mixer", psi_shp, thetas_shp, "ws"),
        ("Std_Relax_XY_Mixer", psi_std, None, "xy"), # Biased Init + XY Mixer
        ("Shp_Relax_XY_Mixer", psi_shp, None, "xy"),
    ]
    
    for p in [1, 2, 3]:
        for name, psi, thetas, mixer in modes:
            print(f"Running {name} p={p}...")
            # optimize_angles uses COBYLA
            params, val = optimize_angles(psi, energies, N, K, mixer, p, thetas)
            
            # Evolve and check overlap
            psi_final = evolve_state(psi, energies, N, params, mixer=mixer, T=1, thetas=thetas)
            overlap = compute_overlap(psi_final, z_opt)
            gap = val - emin
            
            results.append({
                "Method": name,
                "p": p,
                "Energy": val,
                "Gap": gap,
                "Overlap": overlap
            })
            print(f"  Gap: {gap:.6f}, Overlap: {overlap:.4f}")

    # Gradient ADAPT from Biased Initialization
    print("\n--- Gradient ADAPT (Biased Init) ---")
    # Using Standard Relaxation Bias
    pairs = build_pairs(N, "ring")
    # Corrected function call: pool=pairs -> pairs=pairs
    theta_adapt, ops, best_adapt, layers, gates, trace = adapt_qaoa_pairs_gradient(
        psi_std, energies, N, K, max_layers=5, pairs=pairs, T=1
    )
    psi_adapt = evolve_state_ops(psi_std, energies, N, theta_adapt, ops, T=1)
    ov_adapt = compute_overlap(psi_adapt, z_opt)
    gap_adapt = best_adapt - emin
    
    results.append({
        "Method": "Gradient_ADAPT_Biased",
        "p": layers,
        "Energy": best_adapt,
        "Gap": gap_adapt,
        "Overlap": ov_adapt
    })
    print(f"  Layers: {layers}, Gap: {gap_adapt:.6f}, Overlap: {ov_adapt:.4f}")

    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv("results/ws_optimization_results.csv", index=False)
    print("\nResults saved to results/ws_optimization_results.csv")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    for method in df["Method"].unique():
        subset = df[df["Method"] == method]
        # Sort by p just in case
        subset = subset.sort_values("p")
        plt.plot(subset["p"], subset["Gap"], marker='o', label=method)
    
    plt.xlabel("Depth p (or Layers)")
    plt.ylabel("Energy Gap")
    plt.title("Optimized Warm-Start Performance (N=12)")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig("results/plots/optimized_ws_comparison.png")
    print("Plot saved to results/plots/optimized_ws_comparison.png")

if __name__ == "__main__":
    run_experiment()
