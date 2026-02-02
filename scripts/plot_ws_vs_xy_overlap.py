import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

TICKERS = ['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA']

def run_case(pipeline, p, warm, mixer):
    res = pipeline.run_standard(
        N=12, K=6, q=0.5, p=p, mixer=mixer, warm_start=warm,
        formulation='mvo', objective='expectation',
        tickers=TICKERS, start='2021-01-01', end='2023-01-01'
    )
    return res.get('overlap', 0.0)

def main():
    os.makedirs('results/plots', exist_ok=True)
    pipe = PortfolioPipeline(seed=25)
    ps = [1,2,3,4,5]
    cold_xy = []
    fake_ws_x = []
    true_ws = []
    for p in ps:
        o1 = run_case(pipe, p, warm=False, mixer='xy')
        cold_xy.append(o1)
        o2 = run_case(pipe, p, warm=True, mixer='x')
        fake_ws_x.append(o2)
        o3 = run_case(pipe, p, warm=True, mixer='xy')
        true_ws.append(o3)
    plt.figure(figsize=(8,5))
    plt.plot(ps, cold_xy, marker='o', label='XY (Cold)')
    plt.plot(ps, fake_ws_x, marker='s', label='Biased Init + X (Standard)')
    plt.plot(ps, true_ws, marker='^', label='WS Mixer (Egger)')
    plt.xlabel('QAOA Layers (p)')
    plt.ylabel('Overlap with Optimal')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(True)
    out = 'results/plots/warm_vs_xy_overlap.png'
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")

if __name__ == '__main__':
    main()
