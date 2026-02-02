Title: Noise-Resilient Portfolio Optimization with WS-ADAPT-QAOA

Summary:
- We build a constraint-aware QAOA module for portfolio selection using XY mixers operating in the K-hot subspace.
- We integrate Warm-Start from convex relaxation, CVaR objective for tail-risk robustness, and an adaptive ansatz (ADAPT-QAOA).
- Benchmarks show improved energy gaps and overlap versus standard QAOA under realistic noise and correlated returns.

Highlights:
- Constraint-aware XY mixer on feasible subspace to enforce fixed budget.
- Warm-Start initialization leveraging quadratic programming relaxation.
- CVaR objective replacing expectation for NISQ robustness.
- ADAPT-QAOA grows layers from an operator pool to fit the instance.

Experiments:
- Standard vs Upgraded Module comparisons across N in {12,16,20}.
- Noise robustness sweeps with bit-flip channel.
- Advanced formulations: MAD and transaction costs baselines.

Try It:
- Run scripts/orchestrate_experiments.py to reproduce results.
- Explore Adapt QAOA.ipynb for a step-by-step WS-ADAPT-QAOA tutorial.
