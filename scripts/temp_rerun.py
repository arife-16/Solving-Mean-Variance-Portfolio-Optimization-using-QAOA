import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def main():
    pp = PortfolioPipeline(seed=10)
    res = pp.run_standard(
        N=12, K=6, q=0.5, p=1, mixer='xy', warm_start=False,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA'],
        start='2021-01-01', end='2025-12-31'
    )
    print('OK', res['best_energy'], res['energy_gap'])

if __name__ == '__main__':
    main()
