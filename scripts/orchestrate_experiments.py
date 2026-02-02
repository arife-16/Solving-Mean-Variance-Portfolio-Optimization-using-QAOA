
import os
import sys
import csv
import time
import pandas as pd
from itertools import product

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_experiment(pipeline, config, writer):
    """
    Runs a single experiment configuration and writes to CSV.
    """
    mode = config.get('mode', 'standard')
    N = config.get('N', 6)
    K = config.get('K', N // 2)
    p = config.get('p', 1)  # Layers for Standard, max_layers for ADAPT
    mixer = config.get('mixer', 'xy')
    warm = config.get('warm_start', False)
    form = config.get('formulation', 'mvo')
    shots = config.get('shots', 0)
    noise_p = config.get('noise_p', 0.0)
    noise_model = config.get('noise_model', 'depolarizing')
    penalty = config.get('penalty', 100.0)
    seed = config.get('seed', 12345)
    tickers = config.get('tickers')
    start = config.get('start')
    end = config.get('end')
    objective = config.get('objective', 'expectation')
    solver = config.get('solver', 'miqp')
    
    # Update pipeline seed
    pipeline.seed = seed
    
    print(f"Running: N={N}, Seed={seed}, Mode={mode}, Mixer={mixer}, Warm={warm}, Noise={noise_p}")
    
    try:
        if mode == 'standard':
            res = pipeline.run_standard(
                N=N, K=K, q=0.5, p=p, mixer=mixer, warm_start=warm,
                formulation=form, shots=shots, noise_p=noise_p, noise_model=noise_model,
                solver=solver, penalty=penalty, tickers=tickers, start=start, end=end, objective=objective
            )
            layers = p
        else:
            res = pipeline.run_adapt(
                N=N, K=K, q=0.5, max_layers=p, mixer=mixer, warm_start=warm,
                formulation=form, pool='pairs', shots=shots, noise_p=noise_p, noise_model=noise_model,
                penalty=penalty, tickers=tickers, start=start, end=end, objective=objective
            )
            layers = res['layers']
            
        row = [
            seed, mode, N, K, p, mixer, int(warm), form,
            res['best_energy'], res['optimal_energy'], res['energy_gap'],
            res['cvar'], res.get('overlap', 0.0),
            res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'],
            layers, res['duration_sec'], res.get('shots', 0), res.get('noise_p', 0.0)
        ]
        writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error running config {config}: {e}")
        return False

def main():
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/comprehensive_results.csv'
    
    headers = [
        'seed', 'mode', 'N', 'K', 'p_max_layers', 'mixer', 'warm_start', 'formulation',
        'best_energy', 'optimal_energy', 'energy_gap', 'cvar', 'overlap',
        'gate_single', 'gate_two', 'layers_used', 'duration_sec', 'shots', 'noise_p'
    ]
    
    pipeline = PortfolioPipeline()
    seeds = [10, 20, 30, 40, 50]
    # Historical data configuration
    tickers_pool = [
        'AAPL','MSFT','GOOGL','AMZN','META','TSLA','NVDA','JPM','V','UNH',
        'HD','MA','PG','BAC','XOM','PFE','KO','DIS','PEP','CSCO',
        'NFLX','ADBE','INTC','NKE','CRM','ABBV','TMO','AVGO','ORCL','ACN',
        'COST','WMT','MCD','AMD','QCOM','TXN','LIN','CVX','MRK','AMAT'
    ]
    start = '2021-01-01'
    end = '2025-12-31'
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # 1. Scaling: N vs Energy Gap (Standard vs ADAPT)
        print("\n--- Experiment 1: Scaling on Correlated Data (Multi-Seed) ---")
        for N in [8, 12, 16, 20, 22, 24]:
            for mode in ['standard', 'adapt']:
                # For N >= 22, use fewer seeds to save time
                if N >= 22:
                    current_seeds = seeds[:1]
                elif N == 20:
                    current_seeds = seeds[:3]
                else:
                    current_seeds = seeds
                
                for seed in current_seeds:
                    cfg = {
                        'N': N, 'mode': mode, 'p': 2, 'mixer': 'xy', 'warm_start': False, 'seed': seed,
                        'tickers': tickers_pool[:N], 'start': start, 'end': end, 'objective': 'expectation'
                    }
                    run_experiment(pipeline, cfg, writer)
                
        # 2. Warm Start: N=20 (Increased from 12)
        print("\n--- Experiment 2: Warm Start (Multi-Seed) ---")
        for warm in [False, True]:
            for seed in seeds[:3]: # Limit seeds for deep dives
                cfg = {
                    'N': 20, 'mode': 'standard', 'p': 1, 'mixer': 'xy', 'warm_start': warm, 'seed': seed,
                    'tickers': tickers_pool[:20], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg, writer)
            
        # 3. Mixers: N=16 (Increased from 10)
        print("\n--- Experiment 3: Mixers (Multi-Seed) ---")
        for mixer in ['x', 'xy', 'qampa']:
            for seed in seeds[:3]:
                cfg = {
                    'N': 16, 'mode': 'standard', 'p': 1, 'mixer': mixer, 'warm_start': False, 'penalty': 100.0, 'seed': seed,
                    'tickers': tickers_pool[:16], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg, writer)
            
        # 4. Noise Robustness: N=12 (Increased from 10)
        print("\n--- Experiment 4: Noise Robustness (Multi-Seed) ---")
        for np_val in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25]:
            for seed in seeds[:3]: 
                cfg = {
                    'N': 12, 'mode': 'standard', 'p': 1, 'mixer': 'xy', 
                    'warm_start': False, 'shots': 1024, 'noise_p': np_val, 'noise_model': 'bitflip', 'seed': seed,
                    'tickers': tickers_pool[:12], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg, writer)
        
        # 5. Upgraded Module Benchmark: Standard vs ADAPT + Warm + CVaR + XY
        print("\n--- Experiment 5: Upgraded Module vs Standard ---")
        for N in [12, 16, 20, 22, 24]:
            for seed in seeds[:3]:
                # Standard
                cfg_std = {
                    'N': N, 'mode': 'standard', 'p': 2, 'mixer': 'xy', 'warm_start': False, 'formulation': 'mvo', 'seed': seed,
                    'tickers': tickers_pool[:N], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg_std, writer)
                # Adapt Cold vs Warm comparisons (objective expectation and cvar)
                cfg_adapt_cold = {
                    'N': N, 'mode': 'adapt', 'p': 3, 'mixer': 'xy', 'warm_start': False, 'formulation': 'mvo', 'seed': seed,
                    'tickers': tickers_pool[:N], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg_adapt_cold, writer)
                cfg_adapt_warm = {
                    'N': N, 'mode': 'adapt', 'p': 3, 'mixer': 'xy', 'warm_start': True, 'formulation': 'mvo', 'seed': seed,
                    'tickers': tickers_pool[:N], 'start': start, 'end': end, 'objective': 'expectation'
                }
                run_experiment(pipeline, cfg_adapt_warm, writer)
                cfg_adapt_warm_cvar = {
                    'N': N, 'mode': 'adapt', 'p': 3, 'mixer': 'xy', 'warm_start': True, 'formulation': 'mvo', 'seed': seed,
                    'tickers': tickers_pool[:N], 'start': start, 'end': end, 'objective': 'cvar'
                }
                run_experiment(pipeline, cfg_adapt_warm_cvar, writer)
        
        # 6. Advanced Problem Benchmarks: MAD and Transaction Costs
        print("\n--- Experiment 6: Advanced Problems (MAD, Transaction Costs) ---")
        for form in ['mad', 'mvo_tc']:
            for seed in seeds[:3]:
                cfg = {
                    'N': 12, 'mode': 'standard', 'p': 1, 'mixer': 'xy', 'warm_start': False, 'formulation': form, 'seed': seed,
                    'tickers': tickers_pool[:12], 'start': start, 'end': end, 'objective': 'expectation'
                }
                if form == 'mad':
                    cfg['solver'] = 'lp'
                run_experiment(pipeline, cfg, writer)

    print(f"\nExperiments completed. Results saved to {out_csv}")

if __name__ == '__main__':
    main()
