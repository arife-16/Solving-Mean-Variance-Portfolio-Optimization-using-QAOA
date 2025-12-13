import os
import sys
import csv
import json
import argparse
from itertools import product

sys.path.append(os.getcwd())
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def parse_list_int(s):
    return [int(x) for x in s.split(',') if x]

def parse_list_float(s):
    return [float(x) for x in s.split(',') if x]

def parse_list_str(s):
    return [x for x in s.split(',') if x]

def parse_list_bool(s):
    return [x in ('1','true','True') for x in s.split(',') if x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--N_list', type=str, default='6')
    ap.add_argument('--K_list', type=str, default='3')
    ap.add_argument('--p_list', type=str, default='1,2')
    ap.add_argument('--mixers', type=str, default='xy,x')
    ap.add_argument('--warm_list', type=str, default='0,1')
    ap.add_argument('--formulations', type=str, default='mvo,mad,mvo_tc')
    ap.add_argument('--modes', type=str, default='standard,adapt')
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--lam_tc', type=float, default=0.1)
    ap.add_argument('--q', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--samples', type=int, default=32)
    ap.add_argument('--refine_iters', type=int, default=20)
    ap.add_argument('--refine_step', type=float, default=0.05)
    ap.add_argument('--T', type=int, default=1)
    ap.add_argument('--shots', type=int, default=0)
    ap.add_argument('--noise_p', type=float, default=0.0)
    ap.add_argument('--solver', type=str, choices=['bruteforce','milp'], default='bruteforce')
    ap.add_argument('--pairs', type=str, choices=['ring','all'], default='ring')
    ap.add_argument('--out_csv', type=str, default='results/sweep.csv')
    args = ap.parse_args()

    Ns = parse_list_int(args.N_list)
    Ks = parse_list_int(args.K_list)
    Ps = parse_list_int(args.p_list)
    mixers = parse_list_str(args.mixers)
    warms = parse_list_bool(args.warm_list)
    forms = parse_list_str(args.formulations)
    modes = parse_list_str(args.modes)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    pipe = PortfolioPipeline(seed=args.seed)

    headers = ['mode','N','K','p_or_layers','mixer','warm_start','formulation','alpha','lam_tc','best_energy','optimal_energy','energy_gap','cvar','overlap','gate_single','gate_two','layers','duration_sec','solver_used','shots','noise_p','pairs']
    rows = []
    for mode, N, K, p, mixer, warm, form in product(modes, Ns, Ks, Ps, mixers, warms, forms):
        if mode == 'standard':
            res = pipe.run_standard(N=N, K=K, q=args.q, p=p, mixer=mixer, T=args.T, warm_start=warm, alpha=args.alpha, samples=args.samples, refine_iters=args.refine_iters, refine_step=args.refine_step, formulation=form, lam_tc=args.lam_tc, shots=args.shots, noise_p=args.noise_p, solver=args.solver)
            rows.append([
                mode, N, K, p, mixer, int(warm), form, args.alpha, args.lam_tc, res['best_energy'], res['optimal_energy'], res['energy_gap'], res['cvar'], res.get('overlap',''), res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'], '', res['duration_sec'], res.get('solver_used',''), res.get('shots',''), res.get('noise_p',''), ''
            ])
        else:
            res = pipe.run_adapt(N=N, K=K, q=args.q, max_layers=p, mixer=mixer, T=args.T, warm_start=warm, alpha=args.alpha, formulation=form, lam_tc=args.lam_tc, pool='pairs', shots=args.shots, noise_p=args.noise_p, pairs_mode=args.pairs)
            rows.append([
                mode, N, K, p, mixer, int(warm), form, args.alpha, args.lam_tc, res['best_energy'], res['optimal_energy'], res['energy_gap'], res['cvar'], res.get('overlap',''), res['gate_counts']['single_qubit'], res['gate_counts']['two_qubit'], res['layers'], res['duration_sec'], '', res.get('shots',''), res.get('noise_p',''), res.get('pairs_mode','')
            ])

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            w.writerow(r)

    print(json.dumps({"count": len(rows), "out": args.out_csv}, indent=2))

if __name__ == '__main__':
    main()
