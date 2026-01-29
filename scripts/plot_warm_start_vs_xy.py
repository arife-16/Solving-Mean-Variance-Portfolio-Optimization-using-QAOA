import os, sys
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_case(warm):
    pipe = PortfolioPipeline(seed=21)
    res = pipe.run_standard(
        N=12, K=6, q=0.5, p=2, mixer='xy', warm_start=warm,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    return res

def main():
    os.makedirs('results/plots', exist_ok=True)
    cold = run_case(False)
    warm = run_case(True)
    labels = ['XY (Cold)', 'WS Mixer (Warm)']
    gaps = [cold['energy_gap'], warm['energy_gap']]
    plt.figure(figsize=(7,5))
    plt.bar(labels, gaps, color=['steelblue','darkorange'])
    plt.ylabel('Energy Gap')
    plt.title('Warm-Start Mixer vs XY (N=12, Real Data)')
    for i, v in enumerate(gaps):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    out = 'results/plots/warm_vs_xy_N12.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
