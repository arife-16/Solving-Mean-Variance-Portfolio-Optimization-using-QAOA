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

## Biased Initialization (formerly "Warm-Start")
- We solve a QP relaxation for MVO to produce continuous weights and use them to bias the initial state when `--warm_start` is set.
- This is a biased initialization; mixer Hamiltonian is not modified to have the biased state as its ground state.
- Future work: implement true Warm-Start QAOA mixers following Egger et al. so the dynamics rotate around the biased state rather than scattering it.

## Mixers
- `x`: RX all-qubit mixer.
- `xy`: Trotterized ring XY mixer.
- `qampa`: XY mixer with per-pair phased angles scaled by target cardinality.

## Outputs
- JSON results written to `--out`, including energies, CVaR, overlap, parameters, gate counts, duration, solver info.

## Notes
- Commercial solvers are optional; installation enables stronger MIQP baselines.
- For reproducibility, synthetic mode can be used across environments.

## Limitations
- Real-data overlap can be low without error mitigation; current results reflect worst-case unmitigated performance.
- ADAPT-QAOA uses simple operator selection and parameter search; gradient-based scoring and stronger optimizers are future work.
- Warm-start in subspace relies on relaxed QP weights and may not universally improve performance on historical data.
- Noise modeling is simplified (bitflip/depolarizing for shots); no readout error mitigation layer is integrated yet.
- Large-N ADAPT uses K-hot subspace; runtime grows with combinatorial basis size and may be heavy for high N and K.
- Covariance on historical returns uses np.cov with clipping/cleansing; still sensitive to extreme market events.
- Theoretical hardware error accumulation differs from simplified models; see results/noise_comparison.csv for comparison.
- Required shots for 99% confidence are added in results/large_scale_results.csv to contextualize low overlaps.
