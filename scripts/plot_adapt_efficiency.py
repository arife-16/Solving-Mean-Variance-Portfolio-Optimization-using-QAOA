import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline
from quant_portfolio.adapt_qaoa import adapt_qaoa_pairs, build_pairs
from quant_portfolio.qaoa_core import dicke_state
from quant_portfolio.formulations import energies_full

def run_standard(pipeline, p):
    res = pipeline.run_standard(
        N=12, K=6, q=0.5, p=p, mixer='xy', warm_start=False,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    return res['gate_counts']['two_qubit'], res['best_energy'] - res['optimal_energy']

def run_adapt_energy():
    pipe = PipelineWrapperSeed(23)
    prob = pipe.problem()
    energies = energies_full(prob['means'], prob['cov'], 0.5, 12, K=6, penalty=100.0)
    psi0 = dicke_state(12, 6)
    pairs = build_pairs(12, 'ring')
    theta, ops, best, layers, gates, trace = adapt_qaoa_pairs(psi0, energies, 12, 6, 6, pairs, T=1, objective='expectation', alpha=0.2)
    return gates['two_qubit'], float(best - energies.min())

def run_adapt_gradient(pipeline):
    res = pipeline.run_adapt(
        N=12, K=6, q=0.5, max_layers=6, mixer='xy', warm_start=False,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    return res['gate_counts']['two_qubit'], res['best_energy'] - res['optimal_energy']

class PipelineWrapperSeed:
    def __init__(self, seed):
        self.pipe = PortfolioPipeline(seed=seed)
    def problem(self):
        return self.pipe._get_problem(
            N=12, K=6, q=0.5,
            tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
            start='2021-01-01', end='2023-01-01'
        )

def main():
    os.makedirs('results/plots', exist_ok=True)
    pipeline = PortfolioPipeline(seed=23)
    xs_std = []
    ys_std = []
    for p in [1,2,3,4,5]:
        x, y = run_standard(pipeline, p)
        xs_std.append(x)
        ys_std.append(y)
    x_e, y_e = run_adapt_energy()
    x_g, y_g = run_adapt_gradient(pipeline)
    plt.figure(figsize=(8,5))
    plt.plot(xs_std, ys_std, marker='o', label='Standard QAOA')
    plt.scatter([x_e], [y_e], marker='s', label='Energy-ADAPT')
    plt.scatter([x_g], [y_g], marker='^', label='Gradient-ADAPT')
    plt.xscale('linear')
    plt.yscale('log')
    plt.xlabel('Two-Qubit Gate Count')
    plt.ylabel('Energy Gap (E - Eopt)')
    plt.legend()
    out = 'results/plots/adapt_efficiency.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
