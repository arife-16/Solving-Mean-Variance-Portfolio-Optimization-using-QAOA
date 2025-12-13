import os
import sys
import json
import argparse
sys.path.append(os.getcwd())
from quant_portfolio.portfolio_pipeline import PortfolioPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--q", type=float, default=0.5)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--mixer", type=str, default="xy")
    parser.add_argument("--warm_start", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--refine_iters", type=int, default=50)
    parser.add_argument("--refine_step", type=float, default=0.05)
    parser.add_argument("--mode", type=str, choices=["standard", "adapt"], default="standard")
    parser.add_argument("--max_layers", type=int, default=6)
    parser.add_argument("--formulation", type=str, choices=["mvo", "mad", "mvo_tc"], default="mvo")
    parser.add_argument("--lam_tc", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="results/portfolio_experiments.json")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pipe = PortfolioPipeline(seed=1)
    if args.mode == "standard":
        res = pipe.run_standard(N=args.N, K=args.K, q=args.q, p=args.p, mixer=args.mixer, warm_start=args.warm_start, alpha=args.alpha, samples=args.samples, refine_iters=args.refine_iters, refine_step=args.refine_step, formulation=args.formulation, lam_tc=args.lam_tc)
    else:
        res = pipe.run_adapt(N=args.N, K=args.K, q=args.q, max_layers=args.max_layers, mixer=args.mixer, warm_start=args.warm_start, alpha=args.alpha, formulation=args.formulation, lam_tc=args.lam_tc)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
