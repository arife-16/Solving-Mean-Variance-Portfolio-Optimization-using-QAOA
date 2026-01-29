import os
import sys
import csv
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_large_scale_experiment(pipeline, config, writer):
    N = config.get('N')
    K = config.get('K')
    seed = config.get('seed', 42)
    p = config.get('p', 1)
    pipeline.seed = seed
    np.random.seed(seed)
    print(f"Subset Large Scale: N={N}, K={K}, Seed={seed}, p={p}")
    try:
        res = pipeline.run_standard(
            N=N, K=K, q=0.5, p=p, mixer='xy', warm_start=False,
            formulation='mvo', shots=0, noise_p=0.0, noise_model='none',
            solver='bruteforce', penalty=100.0,
            samples=10, refine_iters=10
        )
        row = [
            seed, N, K, p,
            res['best_energy'], res['optimal_energy'], res['energy_gap'],
            res.get('overlap', 0.0), res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'],
            res['duration_sec'], res.get('solver_used','')
        ]
        writer.writerow(row)
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/large_scale_results.csv'
    headers = ['seed','N','K','p','best_energy','optimal_energy','energy_gap','overlap','gate_single','gate_two','duration_sec','solver_used']
    pipeline = PortfolioPipeline()
    seeds = [42, 43, 44]
    configs = [{'N': 24, 'K': 12, 'p': 1}, {'N': 28, 'K': 14, 'p': 1}, {'N': 30, 'K': 15, 'p': 1}]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for cfg in configs:
            for sd in seeds:
                cc = cfg.copy()
                cc['seed'] = sd
                ok = run_large_scale_experiment(pipeline, cc, writer)
                if not ok:
                    break

if __name__ == '__main__':
    main()
