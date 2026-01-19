# Technical Glossary and Mechanisms — Quantum Portfolio Codebase

## Core Concepts
- QAOA
  - A variational quantum algorithm that alternates a cost-phase separator and a mixer to prepare a state whose measurement minimises the problem energy.
  - Mechanism: Applies phases `exp(-i γ_t H)` followed by mixer unitaries with angles `β_t` across `t=1..p`.
  - Code reference: `quant_portfolio/qaoa_core.py:225` (expectation objective implementation).
- ADAPT-QAOA
  - An adaptive variant that grows the ansatz layer-by-layer, selecting operators from a pool to improve the objective.
  - Mechanism: Iteratively samples candidate operator additions and accepts those that decrease energy; records layers and gate counts.
  - Code reference: `quant_portfolio/adapt_qaoa.py:4` (baseline), `quant_portfolio/adapt_qaoa.py:25` (pairwise pool).
- Diagonal Hamiltonian / Energy Precomputation
  - The problem energy is diagonal in computational basis; for bitstring `z`, energy is indexed `E[z]`.
  - Mechanism: Precompute energies for all `z` (or feasible `K`-hot subset) and use elementwise phases during evolution.
  - Code reference: `quant_portfolio/formulations.py:17` (full energies).

## Mixers and Initialization
- X Mixer (RX on all qubits)
  - Applies `Rx(2β)` independently per qubit; mixes amplitudes globally.
  - Mechanism: Tensor contraction of single-qubit `Rx` across all axes.
  - Code reference: `quant_portfolio/qaoa_core.py:66` (apply_rx_all), `quant_portfolio/qaoa_core.py:94` (use in layer).
- XY Ring Mixer (Hard-Constraint)
  - Two-qubit XY interactions along a ring; preserves Hamming weight (K-hot subspace).
  - Mechanism: Trotter steps applying `exp(-i β (X⊗X + Y⊗Y))` to disjoint pairs, then alternating offsets; includes wrap-around `(N-1,0)`.
  - Code reference: `quant_portfolio/qaoa_core.py:73` (ring application), `quant_portfolio/qaoa_core.py:39` (XY unitary).
- Pairwise XY Operators (ADAPT pool)
  - Adds specific `(i,j)` XY gates per layer for fine-grained control; tracks two-qubit counts precisely.
  - Mechanism: `apply_xy_pair` for given pair; ADAPT selects pairs that most reduce energy.
  - Code reference: `quant_portfolio/qaoa_core.py:47` (pair unitary application), `quant_portfolio/adapt_qaoa.py:25` (pairwise growth).
- Dicke State (K-Hot Initialization)
  - Uniform superposition over all bitstrings with exactly `K` ones; seeds feasible subspace.
  - Mechanism: Fill amplitudes at indices with `popcount(z) == K`, normalize.
  - Code reference: `quant_portfolio/qaoa_core.py:3` (Dicke construction).
- Warm-Start State
  - Biases feasible amplitudes using logits derived from expected returns `μ`; improves convergence.
  - Mechanism: Compute per-asset weight `w_i = logit(σ(μ_i))`, exponentiate sums over selected assets for each feasible bitstring, normalize.
  - Code reference: `quant_portfolio/qaoa_core.py:16` (warm-start).

## Objectives and Metrics
- Expectation Objective
  - Minimises `⟨H⟩ = ∑_z p(z) E[z]` where `p(z)` comes from the evolved state.
  - Mechanism: Evolve with cost-phase and mixer; compute `probs * energies` sum.
  - Code reference: `quant_portfolio/qaoa_core.py:225` (implementation), used in `quant_portfolio/portfolio_pipeline.py:37`.
- CVaR (Conditional Value at Risk)
  - Focuses on the worst-tail energies by averaging the top `α` mass of outcomes (noise-resilient objective).
  - Mechanism: Sort energies descending, accumulate probability mass until `α` is reached, average corresponding energies.
  - Code reference: `quant_portfolio/qaoa_core.py:235` (exact), `quant_portfolio/qaoa_core.py:164` (shot-based).
- Overlap with Optimal
  - Probability of measuring the optimal bitstring under current state/noise/shots.
  - Mechanism: Evolve to final `ψ`, compute `|ψ[z_opt]|^2`, optionally with depolarizing and shot sampling.
  - Code reference: `quant_portfolio/qaoa_core.py:215` (compute_overlap), used in `quant_portfolio/portfolio_pipeline.py:66` and `94/106`.
- Gate Counts (CNOT proxy)
  - Reports approximate two-qubit operation counts per configuration; used as CNOT proxies.
  - Mechanism: For X mixer, counts single-qubit ops; for XY ring, counts pair applications per Trotter step and per layer.
  - Code reference: `quant_portfolio/qaoa_core.py:145` (depolarizing), `quant_portfolio/qaoa_core.py:144` context; `quant_portfolio/qaoa_core.py:...` gate count helper in pipeline via `gate_counts` usage `portfolio_pipeline.py:68/105`.

## Noise and Shot Models
- Depolarizing Noise
  - Mixes distribution towards uniform with rate `p`.
  - Mechanism: `probs ← (1−p) probs + p / d`, where `d` is dimension.
  - Code reference: `quant_portfolio/qaoa_core.py:144` (apply_depolarizing).
- Shot-Based Estimation
  - Samples counts from a multinomial over probabilities to estimate expectation/CVaR under finite shots.
  - Mechanism: `counts ∼ Multinomial(shots, probs)`; compute energy averages from counts.
  - Code reference: `quant_portfolio/qaoa_core.py:148` (expectation_shots), `quant_portfolio/qaoa_core.py:164` (cvar_shots).

## Classical Baselines
- Brute Force (K-Hot)
  - Enumerates all bitstrings with exactly `K` ones to find minimum energy.
  - Mechanism: Iterate `z`, check `popcount(z) == K`, evaluate quadratic form.
  - Code reference: `quant_portfolio/classical.py:3` (brute_force_k_hot), `quant_portfolio/classical.py:41` (from energies).
- Local Search
  - Greedy swap-based improvement by exchanging one selected asset with one unselected asset.
  - Mechanism: Evaluate neighbor energies `(i in, j out)` swaps; accept improving moves until none remain.
  - Code reference: `quant_portfolio/classical.py:17` (local_search).
- MILP (McCormick Linearization)
  - Linearizes quadratic binary terms via auxiliary variables `y_ij = x_i x_j` bounded by McCormick constraints.
  - Mechanism: Add constraints `y_ij ≤ x_i`, `y_ij ≤ x_j`, `y_ij ≥ x_i + x_j − 1`; optimize with integer programming.
  - Code reference: `quant_portfolio/mip.py:3` (MILP build and solve).

## Portfolio Formulations
- Mean–Variance Objective (MVO)
  - Energy: `q xᵀΣx − μᵀx` balances risk (variance) vs. return using tradeoff `q`.
  - Mechanism: Quadratic form via Σ and linear term via μ.
  - Code reference: `quant_portfolio/formulations.py:3` (bitstring energy), `quant_portfolio/formulations.py:17` (full energies).
- Mean Absolute Deviation (MAD)
  - Energy penalises absolute deviation from mean portfolio return; robust alternative to variance.
  - Mechanism: Compute portfolio returns `r_p(t)` over horizon and average absolute deviation from mean.
  - Code reference: `quant_portfolio/formulations.py:24` (bitstring MAD), `quant_portfolio/formulations.py:37` (full energies).
- Transaction Costs (MVO+TC)
  - Augments MVO with linear transaction cost term `λ ∑ tc_i x_i`.
  - Mechanism: Add `lam * (tc @ x)` to energy; MILP adapter supports TC inclusion.
  - Code reference: `quant_portfolio/formulations.py:33` (MVO+TC), `quant_portfolio/formulations.py:44` (full energies with TC).

## Pipeline and Benchmarks
- Standard Run
  - Inputs: `(N,K,q,p,mixer,T,warm_start,alpha,shots,noise_p,formulation,solver)`.
  - Mechanism: Build problem; precompute energies; initialize state; sample/refine parameters; compute CVaR and overlap; report metrics.
  - Code reference: `quant_portfolio/portfolio_pipeline.py:27` (run_standard).
- ADAPT Run
  - Inputs: `(N,K,q,max_layers,mixer,T,warm_start,alpha,pool,pairs_mode,shots,noise_p,formulation)`.
  - Mechanism: Build problem; use ring or all-pairs pool; grow layers adaptively; compute CVaR and overlap; report metrics and gate counts.
  - Code reference: `quant_portfolio/portfolio_pipeline.py:72` (run_adapt), `quant_portfolio/adapt_qaoa.py:25` (pairwise).
- Sweep Runner
  - Produces consolidated CSV across parameter grids for mentor review.
  - Mechanism: Iterates combinations of `N,K,p,mixer,warm,formulation,mode`, captures energy/CVaR/overlap/gate counts/solver/shots/noise.
  - Code reference: `benchmarks/run_sweep.py:24` (CLI), `benchmarks/run_sweep.py:58` (CSV rows).

## Advanced Phrases
- Phase Separator
  - The `exp(-i γ_t H)` unitary applying diagonal phases based on energy per basis state.
  - Mechanism: Elementwise multiply state amplitudes by `exp(-i γ E[z])`.
  - Code reference: `quant_portfolio/qaoa_core.py:225` (embedded), explicit phase at `quant_portfolio/qaoa_core.py:229`.
- Trotterization
  - Decomposes non-commuting terms into ordered steps that approximate full evolution.
  - Mechanism: XY ring applies pair gates in alternating patterns per step `T`.
  - Code reference: `quant_portfolio/qaoa_core.py:73` (loop over T steps).
- Operator Pool
  - Set of candidate mixers to add per ADAPT iteration (X, XY-ring, XY-pair).
  - Mechanism: Evaluate each candidate’s objective effect and choose best improvement.
  - Code reference: `quant_portfolio/adapt_qaoa.py:25` (pair pool), `quant_portfolio/qaoa_core.py:99` (ops evolution).
- Two-Qubit Gate Count (CNOT proxy)
  - Counts pairwise operations as a proxy for hardware CNOT usage; aids NISQ benchmarking.
  - Mechanism: Accumulate `(pairs × T × p)` ring gates; ADAPT pairs add `T` per selected pair per layer.
  - Code reference: `quant_portfolio/adapt_qaoa.py:45` (pair gate accounting), `quant_portfolio/portfolio_pipeline.py:68/105` (reporting).
- K-Hot Constraint (Fixed Budget)
  - Feasible subspace restricted to bitstrings with exactly `K` ones, matching fixed-budget selection.
  - Mechanism: Dicke initialization ensures amplitudes lie inside feasible subspace; XY mixers preserve Hamming weight.
  - Code reference: `quant_portfolio/qaoa_core.py:3` (Dicke), `quant_portfolio/qaoa_core.py:73` (XY ring).

## Integration Notes
- Solver Selection
  - `solver='milp'` uses integer programming where available; otherwise brute-force baseline for small sizes.
  - Code reference: `quant_portfolio/portfolio_pipeline.py:57` (MILP path).
- Noise and Shots in Pipeline
  - `shots>0` routes to shot-based objective functions; `noise_p` mixes probabilities via depolarizing.
  - Code reference: `quant_portfolio/portfolio_pipeline.py:37` (shot toggle), `quant_portfolio/qaoa_core.py:144` (noise).

