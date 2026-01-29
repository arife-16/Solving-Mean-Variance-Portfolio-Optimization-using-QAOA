import os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def main():
    os.makedirs('results/plots', exist_ok=True)
    pipe = PortfolioPipeline(seed=12)
    res = pipe.run_adapt(
        N=12, K=6, q=0.5, max_layers=6, mixer='xy', warm_start=True,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2023-01-01'
    )
    trace = res.get('trace', [])
    if trace:
        plt.figure(figsize=(8,5))
        plt.plot(range(len(trace)), trace, marker='o')
        plt.title('ADAPT Convergence (Energy vs Layer) - N=12, xy, warm')
        plt.xlabel('Layer')
        plt.ylabel('Best Energy')
        plt.grid(True)
        out = 'results/plots/adapt_convergence_N12.png'
        plt.savefig(out, dpi=200)
        plt.close()
        print(f"Saved {out}")
    else:
        print("No trace available in result.")

if __name__ == '__main__':
    main()
