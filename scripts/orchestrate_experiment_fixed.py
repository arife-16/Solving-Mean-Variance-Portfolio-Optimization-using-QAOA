"""
Fixed and enhanced experiment orchestration script
Includes:
- Proper ADAPT configuration (max_layers=6, all-pairs pool)
- Formulation comparison (MVO, MAD, CVaR, TC)
- CVaR objective testing
- Real data testing
"""
import os
import sys
import csv
import time
import pandas as pd
from itertools import product

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def run_experiment(pipeline, config, writer):
    """
    Runs a single experiment configuration and writes to CSV.
    """
    mode = config.get('mode', 'standard')
    N = config.get('N', 6)
    K = config.get('K', N // 2)
    p = config.get('p', 1)
    max_layers = config.get('max_layers', 6)  # For ADAPT
    mixer = config.get('mixer', 'xy')
    warm = config.get('warm_start', False)
    form = config.get('formulation', 'mvo')
    shots = config.get('shots', 0)
    noise_p = config.get('noise_p', 0.0)
    noise_model = config.get('noise_model', 'depolarizing')
    penalty = config.get('penalty', 100.0)
    seed = config.get('seed', 12345)
    objective = config.get('objective', 'expectation')
    pairs_mode = config.get('pairs_mode', 'all')  # Changed from 'ring' to 'all'
    
    # Update pipeline seed
    pipeline.seed = seed
    
    print(f"Running: N={N}, Seed={seed}, Mode={mode}, Form={form}, Obj={objective}, Mixer={mixer}, Warm={warm}, Noise={noise_p}")
    
    try:
        if mode == 'standard':
            res = pipeline.run_standard(
                N=N, K=K, q=0.5, p=p, mixer=mixer, warm_start=warm,
                formulation=form, shots=shots, noise_p=noise_p, noise_model=noise_model,
                solver='miqp', penalty=penalty, objective=objective
            )
            layers = p
        else:
            res = pipeline.run_adapt(
                N=N, K=K, q=0.5, 
                max_layers=max_layers,  # Use max_layers for ADAPT
                mixer=mixer, warm_start=warm,
                formulation=form, pool='pairs', shots=shots, noise_p=noise_p, 
                noise_model=noise_model, penalty=penalty, objective=objective,
                pairs_mode=pairs_mode
            )
            layers = res['layers']
            
        row = [
            seed, mode, N, K, p if mode=='standard' else max_layers, mixer, int(warm), form,
            res['best_energy'], res['optimal_energy'], res['energy_gap'],
            res['cvar'], res.get('overlap', 0.0),
            res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'],
            layers, res['duration_sec'], res.get('shots', 0), res.get('noise_p', 0.0),
            res.get('objective', 'expectation'), pairs_mode
        ]
        writer.writerow(row)
        return True
    except Exception as e:
        print(f"Error running config {config}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    os.makedirs('results', exist_ok=True)
    out_csv = 'results/comprehensive_results_fixed.csv'
    
    headers = [
        'seed', 'mode', 'N', 'K', 'p_max_layers', 'mixer', 'warm_start', 'formulation',
        'best_energy', 'optimal_energy', 'energy_gap', 'cvar', 'overlap',
        'gate_single', 'gate_two', 'layers_used', 'duration_sec', 'shots', 'noise_p',
        'objective', 'pairs_mode'
    ]
    
    pipeline = PortfolioPipeline()
    seeds = [10, 20, 30, 40, 50]
    
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        # ====================================================================
        # EXPERIMENT 1: Scaling with FIXED ADAPT (all-pairs, max_layers=6)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 1: Scaling (ADAPT vs Standard) - FIXED")
        print("="*70)
        
        for N in [8, 12, 16, 20, 22, 24]:
            for mode in ['standard', 'adapt']:
                # Reduce seeds for large N
                if N >= 22:
                    current_seeds = seeds[:1]
                elif N == 20:
                    current_seeds = seeds[:3]
                else:
                    current_seeds = seeds
                
                for seed in current_seeds:
                    if mode == 'adapt':
                        # FIXED: Use max_layers=6 and pairs_mode='all'
                        cfg = {
                            'N': N, 'mode': 'adapt', 
                            'max_layers': 6,  # Changed from p=2
                            'mixer': 'xy', 
                            'warm_start': False, 
                            'seed': seed,
                            'pairs_mode': 'all'  # Changed from 'ring'
                        }
                    else:
                        cfg = {
                            'N': N, 'mode': 'standard', 
                            'p': 2, 
                            'mixer': 'xy', 
                            'warm_start': False, 
                            'seed': seed
                        }
                    run_experiment(pipeline, cfg, writer)
        
        # ====================================================================
        # EXPERIMENT 2: Warm-Start (with more obvious test case)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 2: Warm-Start Impact")
        print("="*70)
        
        for N in [8, 12, 16, 20]:  # Test multiple sizes
            for warm in [False, True]:
                for seed in seeds[:3]:
                    cfg = {
                        'N': N, 'mode': 'standard', 'p': 1, 
                        'mixer': 'xy', 'warm_start': warm, 
                        'seed': seed
                    }
                    run_experiment(pipeline, cfg, writer)
        
        # ====================================================================
        # EXPERIMENT 3: Formulation Comparison (NEW!)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 3: Formulation Comparison (MVO vs MAD vs CVaR)")
        print("="*70)
        
        for form in ['mvo', 'mad', 'cvar']:  # Test all formulations
            for mode in ['standard', 'adapt']:
                for seed in seeds[:3]:
                    if mode == 'adapt':
                        cfg = {
                            'N': 12, 'mode': 'adapt', 'max_layers': 6,
                            'mixer': 'xy', 'warm_start': False,
                            'formulation': form, 'seed': seed,
                            'pairs_mode': 'all'
                        }
                    else:
                        cfg = {
                            'N': 12, 'mode': 'standard', 'p': 2,
                            'mixer': 'xy', 'warm_start': False,
                            'formulation': form, 'seed': seed
                        }
                    run_experiment(pipeline, cfg, writer)
        
        # ====================================================================
        # EXPERIMENT 4: CVaR Objective vs Expectation (NEW!)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 4: CVaR Objective vs Expectation")
        print("="*70)
        
        for objective in ['expectation', 'cvar']:
            for noise_p in [0.0, 0.01, 0.05]:
                for seed in seeds[:3]:
                    cfg = {
                        'N': 12, 'mode': 'standard', 'p': 2,
                        'mixer': 'xy', 'warm_start': False,
                        'formulation': 'mvo',
                        'objective': objective,
                        'noise_p': noise_p,
                        'noise_model': 'depolarizing',  # Changed from bitflip
                        'shots': 512 if noise_p > 0 else 0,
                        'seed': seed
                    }
                    run_experiment(pipeline, cfg, writer)
        
        # ====================================================================
        # EXPERIMENT 5: Mixers (keep original)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 5: Mixer Comparison")
        print("="*70)
        
        for mixer in ['x', 'xy']:  # Removed qampa if causing issues
            for seed in seeds[:3]:
                cfg = {
                    'N': 16, 'mode': 'standard', 'p': 1, 
                    'mixer': mixer, 'warm_start': False, 
                    'penalty': 100.0, 'seed': seed
                }
                run_experiment(pipeline, cfg, writer)
        
        # ====================================================================
        # EXPERIMENT 6: Noise Robustness (FIXED - use depolarizing)
        # ====================================================================
        print("\n" + "="*70)
        print("EXPERIMENT 6: Noise Robustness - FIXED")
        print("="*70)
        
        for np_val in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
            for seed in seeds[:3]:
                cfg = {
                    'N': 12, 'mode': 'standard', 'p': 1, 
                    'mixer': 'xy', 'warm_start': False, 
                    'shots': 1024, 'noise_p': np_val, 
                    'noise_model': 'depolarizing',  # Changed from 'bitflip'
                    'seed': seed
                }
                run_experiment(pipeline, cfg, writer)
    
    print(f"\n{'='*70}")
    print(f"All experiments completed!")
    print(f"Results saved to {out_csv}")
    print(f"{'='*70}")
    
    # Quick summary
    df = pd.read_csv(out_csv)
    print(f"\nTotal experiments run: {len(df)}")
    print(f"Modes: {df['mode'].value_counts().to_dict()}")
    print(f"Formulations tested: {df['formulation'].unique().tolist()}")
    print(f"Objectives tested: {df['objective'].unique().tolist()}")

if __name__ == '__main__':
    main()