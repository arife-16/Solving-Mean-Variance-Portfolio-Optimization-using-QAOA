# Solving Mean-Variance Portfolio Optimization using QAOA

## Overview
- End-to-end stack for portfolio optimization using Standard and ADAPT-QAOA.
- Real data pipeline via tickers with historical fetch or CSV inputs.
- Classical baselines: MILP/MIQP via cvxpy/pulp, with QP relaxation fallback.
- Mixers: X, XY-ring, and QAMPA variant.
- Objectives: Expectation and CVaR, with shot-based local noise.

## Setup
- Python 3.9+
- Recommended: `python3 -m venv .venv && source .venv/bin/activate`
- Install:
  - `pip install numpy pandas cvxpy yfinance pulp`
  - Optional commercial solvers for cvxpy: GUROBI, MOSEK, CPLEX

## Data Options
- Real fetch:
  - `--tickers "AAPL,MSFT,GOOG" --start 2020-01-01 --end 2021-01-01`
- CSV inputs:
  - `--prices_csv path/to/prices.csv` (rows time, cols assets)
  - `--tc_csv path/to/transaction_costs.csv` (vector length N)
- If neither is provided, synthetic returns are generated.

## Running
- Standard QAOA:
  - `python3 scripts/run_portfolio_experiments.py --mode standard --N 12 --K 6 --q 0.5 --p 2 --mixer qampa --objective cvar --alpha 0.2 --solver miqp --tickers "AAPL,MSFT,GOOG" --start 2020-01-01 --end 2021-01-01`
- ADAPT-QAOA:
  - `python3 scripts/run_portfolio_experiments.py --mode adapt --N 12 --K 6 --q 0.5 --max_layers 6 --mixer qampa --objective expectation --tickers "AAPL,MSFT,GOOG" --start 2020-01-01 --end 2021-01-01`
- Noise:
  - `--shots 2048 --noise_model bitflip --noise_p 0.01`

## Solvers
- MIQP: `--solver miqp` (cvxpy backends: GUROBI, MOSEK, CPLEX, SCIP, ECOS_BB)
- MILP: `--solver milp` (pulp CBC)
- Fallback:
  - If integer solver is unavailable, QP relaxation is solved and rounded.

## Warm-Start
- QP relaxation for MVO is solved to produce continuous weights used to bias the initial state when `--warm_start` is set.

## Mixers
- `x`: RX all-qubit mixer.
- `xy`: Trotterized ring XY mixer.
- `qampa`: XY mixer with per-pair phased angles scaled by target cardinality.

## Outputs
- JSON results written to `--out`, including energies, CVaR, overlap, parameters, gate counts, duration, solver info.

## Notes
- Commercial solvers are optional; installation enables stronger MIQP baselines.
- For reproducibility, synthetic mode can be used across environments.
