# Strategic Roadmap Presentation — Codebase and Plan Alignment

## Executive Summary
- Goal: Deliver a quantum portfolio optimization module with ADAPT-QAOA, hard-constraint XY mixers, warm-starting, and CVaR objectives, benchmarked against classical MIP solvers on realistic and noisy settings.
- Approach: Implement a self-contained, reproducible stack (`quant_portfolio`) while using the QOKit codebase as a technical reference for circuits, Dicke states, diagonal energies, and efficient simulation patterns.
- Status: Internal stack implements Standard QAOA, ADAPT-QAOA with pairwise operator pool, XY mixers, warm-start, CVaR, shot/noise models, MILP baseline, and a sweep runner for consolidated CSV benchmarking.

## Codebase Structure
- `quant_portfolio` (our implementation)
  - Data and preprocessing: `quant_portfolio/data.py`
  - Formulations: `quant_portfolio/formulations.py` (MVO, MAD, MVO+TC energies)
  - Classical baselines: `quant_portfolio/classical.py` (brute-force, local search)
  - QAOA core: `quant_portfolio/qaoa_core.py` (Dicke, warm-start, XY/X mixers, expectation/CVaR, shots/noise)
  - ADAPT engine: `quant_portfolio/adapt_qaoa.py` (greedy layer growth, pairwise XY pool)
  - Pipeline: `quant_portfolio/portfolio_pipeline.py` (Standard/ADAPT runs, overlap, gate counts)
  - MIP adapter: `quant_portfolio/mip.py` (MILP via McCormick linearization using `pulp` if available)
  - Benchmarks: `benchmarks/run_sweep.py` (parameter sweeps, CSV output)
  - Single-run CLI: `scripts/run_portfolio_experiments.py`
- QOKit references
  - Circuits/objectives: portfolio and general objective modules in `qokit` (see `docs/QOKit_codebase_overview.md`)
  - Dicke states, XY mixers, diagonal energies, and fast simulation backends provide design and performance patterns.

## Roadmap Mapping
- Phase 1: Foundations & Classical Benchmarking
  - Data pipeline: synthetic returns for immediate use; replace with historical prices later (`quant_portfolio/data.py`).
  - Preprocessing: μ and Σ computation implemented (`compute_mu_sigma`).
  - Classical MIP baseline: MILP adapter for MVO and MVO+TC (`quant_portfolio/mip.py`), plus brute force for small N (`classical.py`).
- Phase 2: Core Quantum Engine Development
  - Mixers:
    - X mixer (`rx` on all qubits) in `quant_portfolio/qaoa_core.py:66–71`.
    - Hard-constraint XY ring mixer preserving K-hot subspace in `quant_portfolio/qaoa_core.py:73–82`.
  - Standard QAOA:
    - Phase separator and mixer application with parameters `[γ_1..γ_p, β_1..β_p]` in `quant_portfolio/qaoa_core.py:225–233`.
    - Integrated in pipeline `run_standard` (`quant_portfolio/portfolio_pipeline.py:27–70`).
  - ADAPT-QAOA:
    - Baseline greedy growth (`adapt_qaoa`) and pairwise XY operator pool (`adapt_qaoa_pairs`) in `quant_portfolio/adapt_qaoa.py`.
    - Integrated in pipeline `run_adapt` with `pool='pairs'` and `pairs_mode='ring|all'` (`quant_portfolio/portfolio_pipeline.py:72–109`).
- Phase 3: NISQ Performance Enhancements
  - Warm-start:
    - Feasible-state biasing via logits derived from μ (`quant_portfolio/qaoa_core.py:16–33`).
  - CVaR objective:
    - Exact probabilities (`quant_portfolio/qaoa_core.py:235–243`).
    - Shot/noise-based estimates (`quant_portfolio/qaoa_core.py:164–189`).
  - Noisy simulator:
    - Multinomial shots (`qaoa_expectation_shots`) and depolarizing noise model (`apply_depolarizing`) in `quant_portfolio/qaoa_core.py:148–162` and `144–147`.
  - Local search:
    - Classical polishing of bitstrings via swaps (`quant_portfolio/classical.py`).
- Phase 4: Full-Scale Benchmarking & Analysis
  - Sweep runner:
    - Parameter grids across `N,K,p`, mixer, warm-start, formulation, mode; CSV metrics including energy, CVaR, overlap, gate counts (`benchmarks/run_sweep.py:24–81`).
  - Metrics:
    - `best_energy`, `optimal_energy`, `energy_gap`, `cvar`, `overlap`, `gate_single/two`, `layers`, `duration_sec`, `solver_used`, `shots`, `noise_p`, `pairs`.

## Relevance of QOKit Codebase
- Dicke and XY mixer designs:
  - Reference implementation of Dicke state construction and XY ring mixers inform our approach to constraint-aware subspaces and Trotterized XY evolution.
- Objective building over diagonal energies:
  - QOKit’s `energy_qaoa_objective.py` and portfolio modules show how to precompute diagonal costs and compute expectations/overlaps efficiently, mirrored in our `energies_full` and QAOA core.
- Simulator backends:
  - QOKit’s Python/GPU/C simulators demonstrate patterns for scaling beyond naive loops; our current implementation remains NumPy-based but follows similar abstractions and can be extended.
- Parameter utilities:
  - Best-practice scaling, parameterization and mixer-specific adjustments guided our parameter layout and gate-count metrics.

## Demonstration Plan
- Single experiment (Standard QAOA, MVO):
  - Command: `python3 scripts/run_portfolio_experiments.py --mode standard --N 12 --K 6 --q 0.7 --p 2 --mixer xy --alpha 0.2 --out results/standard_mvo.json`
  - Shows energy, CVaR, overlap, gate counts, and runtime.
- ADAPT-QAOA (pairwise pool):
  - Command: `python3 scripts/run_portfolio_experiments.py --mode adapt --N 12 --K 6 --q 0.7 --max_layers 3 --mixer xy --alpha 0.2 --out results/adapt_pairs_mvo.json`
  - Reports layers used, gate counts, CVaR, overlap, and runtime.
- Noisy shot-based evaluation:
  - Sweep: `python3 benchmarks/run_sweep.py --N_list 12 --K_list 6 --p_list 1,2 --mixers xy --warm_list 0,1 --formulations mvo,mvo_tc --modes standard --shots 1024 --noise_p 0.03 --solver milp --out_csv results/standard_mip_noisy.csv`
  - CSV consolidates metrics for mentor review.

## File References
- Pipeline
  - `quant_portfolio/portfolio_pipeline.py:27–70` — Standard QAOA run function.
  - `quant_portfolio/portfolio_pipeline.py:72–109` — ADAPT-QAOA run function.
- QAOA Core
  - `quant_portfolio/qaoa_core.py:3–14` — Dicke state construction.
  - `quant_portfolio/qaoa_core.py:16–33` — Warm-start initial state.
  - `quant_portfolio/qaoa_core.py:73–82` — XY ring mixer.
  - `quant_portfolio/qaoa_core.py:225–233` — Expectation objective.
  - `quant_portfolio/qaoa_core.py:235–243` — CVaR objective (exact).
  - `quant_portfolio/qaoa_core.py:148–162` — Shot-based expectation under noise.
  - `quant_portfolio/qaoa_core.py:164–189` — Shot-based CVaR under noise.
- Benchmarks
  - `benchmarks/run_sweep.py:24–45` — CLI setup for sweeps.
  - `benchmarks/run_sweep.py:58–70` — Writing consolidated CSV metrics.

## Validation and Benchmarks
- Correctness:
  - Compare `best_energy` to `optimal_energy` (MILP or brute-force) and track `energy_gap`.
  - Overlap measures probability mass on optimal solution under noise/shots.
- Performance:
  - Gate counts as proxy for CNOTs; layer-to-convergence measures ADAPT efficiency.
  - CSV artifacts under `results/` to ensure repeatability.

## Risks and Mitigations
- MILP dependencies:
  - If `pulp` or solvers are unavailable, fallback to brute force and note scale limits; plan to integrate `cvxpy` MIQP as an alternative.
- Noisy models:
  - Depolarizing is a simplification; add device-specific channels (bit-flip, phase-flip) as needed.
- Operator pool selection:
  - Current pairwise pool uses ring or all pairs; extend to heuristic scoring of pairs and add pruning to manage growth.

## Next Steps
- Integrate real historical data and transaction fee models.
- Add CVaR-aware optimizer loops (e.g., COBYLA/Nelder–Mead on CVaR directly).
- Expand solver adapters (`cvxpy` MIQP) and device noise models.
- Automate report generation from sweeps (plots and tables).

## Q&A Prep
- Constraint handling:
  - XY mixer over Dicke-initialized states keeps the evolution inside the K-hot feasible subspace.
- ADAPT pool choice:
  - Pairwise XY gates align with hardware-native two-qubit operations and give fine-grained control; gate counts reported per layer.
- CVaR rationale:
  - Focuses optimization on tail-risk under noise, improving robustness vs. pure expectation.

