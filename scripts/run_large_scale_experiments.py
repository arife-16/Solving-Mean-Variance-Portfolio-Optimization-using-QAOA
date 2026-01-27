
import os
import sys
import csv
import time
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_large_scale_experiment(pipeline, config, writer):
    """
    Runs a single large-scale experiment configuration and writes to CSV.
    """
    N = config.get('N')
    K = config.get('K')
    seed = config.get('seed', 42)
    p = config.get('p', 1)
    
    # Update pipeline seed
    pipeline.seed = seed
    np.random.seed(seed)
    
    print(f"Running Large Scale: N={N}, K={K}, Seed={seed}, p={p}")
    
    try:
        # Using standard QAOA (subspace implementation triggered by N > 24)
        res = pipeline.run_standard(
            N=N, K=K, q=0.5, p=p, mixer='xy', warm_start=False,
            formulation='mvo', shots=0, noise_p=0.0, noise_model='none',
            solver='bruteforce', penalty=100.0,
            samples=20, refine_iters=20 # Reduced samples for speed in large scale
        )
        
        row = [
            seed, N, K, p,
            res['best_energy'], res['optimal_energy'], res['energy_gap'],
            res['overlap'], res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'],
            res['duration_sec'], res['solver_used']
        ]
        writer.writerow(row)
        # Flush to ensure data is written immediately
        sys.stdout.flush()
        return True
    except Exception as e:
        print(f"Error running config {config}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/large_scale_results.csv'
    
    headers = [
        'seed', 'N', 'K', 'p',
        'best_energy', 'optimal_energy', 'energy_gap', 'overlap',
        'gate_single', 'gate_two', 'duration_sec', 'solver_used'
    ]
    
    pipeline = PortfolioPipeline()
    seeds = [10, 20] # Two seeds for validation
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        print("\n--- Large Scale Subspace Experiments ---")
        
        # Configurations to run
        # N=30 cases
        configs = [
            {'N': 30, 'K': 2, 'p': 1},
            {'N': 30, 'K': 5, 'p': 1},
            {'N': 30, 'K': 10, 'p': 1}, # ~30M states, might take a while
            
            # N=40 cases
            {'N': 40, 'K': 2, 'p': 1},
            {'N': 40, 'K': 5, 'p': 1},  # ~658k states
        ]
        
        for config in configs:
            for seed in seeds:
                current_config = config.copy()
                current_config['seed'] = seed
                success = run_large_scale_experiment(pipeline, current_config, writer)
                if not success:
                    print(f"Skipping remaining seeds for config {config} due to error.")
                    break

if __name__ == "__main__":
    main()
