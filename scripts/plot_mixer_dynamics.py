import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_series(pipeline, p, warm, mixer):
    res = pipeline.run_standard(
        N=12, K=6, q=0.5, p=p, mixer=mixer, warm_start=warm,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    emin = res['optimal_energy']
    approx_ratio = res['best_energy'] / emin if emin != 0 else float('inf')
    return approx_ratio, res.get('overlap', 0.0)

def main():
    os.makedirs('results/plots', exist_ok=True)
    pipe = PortfolioPipeline(seed=22)
    ps = [1,2,3,4,5]
    cold = []
    fake_ws = []
    true_ws = []
    for p in ps:
        ar_c, _ = run_series(pipe, p, warm=False, mixer='x')
        cold.append(ar_c)
        ar_f, _ = run_series(pipe, p, warm=True, mixer='x')
        fake_ws.append(ar_f)
        ar_t, _ = run_series(pipe, p, warm=True, mixer='xy')
        true_ws.append(ar_t)
    plt.figure(figsize=(8,5))
    plt.plot(ps, cold, marker='o', label='Cold + Standard Mixer (X)')
    plt.plot(ps, fake_ws, marker='s', label='Biased Init + Standard Mixer (X)')
    plt.plot(ps, true_ws, marker='^', label='True Warm-Start Mixer (WS)')
    plt.xlabel('QAOA Layers (p)')
    plt.ylabel('Approximation Ratio (E / Emin)')
    plt.legend()
    out = 'results/plots/mixer_dynamics_validation.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
