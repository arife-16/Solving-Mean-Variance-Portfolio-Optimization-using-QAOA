import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline
from quant_portfolio.adapt_qaoa import adapt_qaoa_pairs

def run_adapt_gradient():
    pipe = PortfolioPipeline(seed=13)
    res = pipe.run_adapt(
        N=12, K=6, q=0.5, max_layers=6, mixer='xy', warm_start=False,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    return res

def run_adapt_energy():
    # energy-based selection using internal function directly
    import numpy as np
    from quant_portfolio.qaoa_core import dicke_state, qaoa_expectation_ops, evolve_state_ops
    from quant_portfolio.adapt_qaoa import build_pairs
    # assemble problem quickly via pipeline internals
    pipe = PortfolioPipeline(seed=13)
    prob = pipe._get_problem(N=12, K=6, q=0.5,
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01')
    energies = prob['means']  # placeholder, but we need full energies via formulations
    from quant_portfolio.formulations import energies_full
    energies = energies_full(prob['means'], prob['cov'], 0.5, 12, K=6, penalty=100.0)
    psi0 = dicke_state(12, 6)
    pairs = build_pairs(12, 'ring')
    theta, ops, best, layers, gates, trace = adapt_qaoa_pairs(psi0, energies, 12, 6, 6, pairs, T=1, objective='expectation', alpha=0.2)
    return {'best_energy': float(best), 'layers': int(layers), 'gate_counts': gates, 'trace': trace}

def main():
    os.makedirs('results/plots', exist_ok=True)
    g = run_adapt_gradient()
    e = run_adapt_energy()
    plt.figure(figsize=(8,5))
    if g.get('trace'):
        plt.plot(range(len(g['trace'])), g['trace'], marker='o', label='Gradient ADAPT')
    if e.get('trace'):
        plt.plot(range(len(e['trace'])), e['trace'], marker='s', label='Energy ADAPT')
    plt.title('ADAPT Convergence: Gradient vs Energy (N=12)')
    plt.xlabel('Layer')
    plt.ylabel('Best Energy')
    plt.legend()
    out = 'results/plots/adapt_gradient_vs_energy_N12.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
