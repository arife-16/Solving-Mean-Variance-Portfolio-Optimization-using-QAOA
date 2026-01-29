import os, sys, csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def main():
    pipe = PortfolioPipeline(seed=5)
    res = pipe.run_adapt(
        N=16, K=8, q=0.5, max_layers=6, mixer='xy', warm_start=True,
        formulation='mvo', objective='expectation',
        tickers=['AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH','HD','MA','PG','BAC','XOM','PFE'],
        start='2021-01-01', end='2025-12-31'
    )
    N_gates = res['gate_counts']['two_qubit']
    epsilon = 0.01
    P_success = (1 - epsilon) ** N_gates
    P_fail = 1 - P_success
    out_path = os.path.join('results','noise_comparison.csv')
    os.makedirs('results', exist_ok=True)
    rows = [['N','mode','layers','two_qubit_gates','epsilon','P_success','P_fail','bitflip_noise_p','modeled_note']]
    rows.append([16,'adapt',res['layers'],N_gates,epsilon,P_success,P_fail,0.01,'Bitflip applies post-measurement flips; hardware error compounds per gate'])
    with open(out_path,'w',newline='') as f:
        w=csv.writer(f)
        w.writerows(rows)
    print(f"Wrote {out_path} with {len(rows)-1} rows")

if __name__ == '__main__':
    main()
