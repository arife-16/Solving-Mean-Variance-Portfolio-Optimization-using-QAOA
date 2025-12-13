# QOKit Enhanced MLMVN — Codebase Overview

This document explains the purpose of every notable file in the folder `QOKit_enhanced_MLMVN-main/`. It is organized by directory with concise, practical descriptions focused on how the code fits together for QAOA workflows, simulators, datasets, and example notebooks.

## Top-Level
- `README.md` — Project overview, usage instructions, and high-level capabilities.
- `LICENSE` — Licensing terms.
- `CODE_OF_CONDUCT.md` — Community conduct guidelines.
- `CONTRIBUTING.md` — Contribution process and standards.
- `.gitignore` — Git ignore patterns.
- `.pre-commit-config.yaml` — Pre-commit hooks configuration.
- `header.txt` — Copyright header template used in sources.
- `pyproject.toml` — Build-system and packaging metadata (Python project configuration).

## QOKit (Examples)
Path: `QOKit_enhanced_MLMVN-main/QOKit/examples/`
- `README.md` — How to run notebooks and example flows.
- `QOKit_general_simulation.ipynb` — A general-purpose QOKit simulation walkthrough.
- `QAOA_portfolio_optimization.ipynb` — QAOA portfolio optimization demo, including problem construction and circuits.
- `QAOA_objective_for_general_Hamiltonian.ipynb` — Shows objective building against general diagonal Hamiltonians.
- `QAOA_iterative_interpolation_SK.ipynb` — SK-model interpolation experiment.
- `QAOA_SK_Expectation_vs_P.ipynb` — SK-model expectation versus layers `p`.
- `QAOA_LABS_circuit_generation.ipynb` — LABS circuits generation examples.
- `QAOA_LABS_optimization.ipynb` — LABS optimization study.
- `advanced/classical_solvers_for_LABS/README.md` — Classical LABS solver readme.
- `advanced/classical_solvers_for_LABS/generate_lp_files.py` — Generates LP/MIP files for LABS instances.
- `advanced/classical_solvers_for_LABS/run_cplex.py` — Runs IBM CPLEX on LABS LPs.
- `advanced/classical_solvers_for_LABS/run_gurobi.py` — Runs Gurobi on LABS LPs.
- `advanced/classical_solvers_for_LABS/LABS_optimal_merit_factors.csv` — Reference optimal merit factors for LABS.
- `assets/C_over_N.npy`, `assets/C_p.npy`, `assets/p_list.npy` — Precomputed assets referenced in example analyses.
- `advanced/compare_maxcut_parameters.ipynb` — Notebook comparing MaxCut parameterizations.
- `advanced/generate_optimized_circuits_for_Np_experiments.ipynb` — Produces optimized circuits for experimental sweeps.

## qokit (Core Library)
Path: `QOKit_enhanced_MLMVN-main/qokit/`
- `__init__.py` — Package initializer exporting library entry points.
- `README.md` (in `qokit/assets/`) — Describes available datasets and precomputations.

### Cython and Compiled Extensions
- `Cpy/qaoa_maxcut_energy.pyx` — Cython accelerated MaxCut energy computations for QAOA objectives.
- `Cpy/setup.py` — Build script for the Cython extension.

### Assets and Datasets
Path: `qokit/assets/`
- `maxcut_datasets/fixed_angles_for_regular_graphs.json` — Parameter tables for regular graphs (MaxCut).
- `precomputed_bitstrings/precomputed_bitstrings_*.npy` — Precomputed optimal or sampled bitstrings by size for fast objective evaluation.
- `precomputed_merit_factors/precomputed_energies_*.npy` — Precomputed energies/merit factors for LABS instances.
- `best_LABS_QAOA_parameters_wrt_MF.json` — Best LABS parameters by merit factor.
- `best_LABS_QAOA_parameters_wrt_overlap.json` — Best LABS parameters by overlap.
- `best_SK_QAOA_parameters.json` — Best SK-model QAOA parameters.
- `__init__.py` — Convenience imports for assets.

### Classical Methods
Path: `qokit/classical_methods/`
- `__init__.py` — Package initializer.
- `generate_lp.py` — LP/MIP formulations generator for classical solvers (e.g., CPLEX/Gurobi).
- `utils.py` — Helpers for constructing and exporting classical problem instances.

### Fast Simulator (“fur”) Stack
Path: `qokit/fur/`
- `__init__.py` — Package initializer and component registration.
- `lazy_import.py` — Deferred imports to avoid heavy dependency load.
- `qaoa_simulator_base.py` — Base interfaces and shared logic for simulator variants.
- `python/energy_qaoa_simulator.py`, `python/qaoa_simulator.py`, `python/fur.py`, `python/gates.py`, `python/utils.py` — Pure-Python fast simulator implementations and gate primitives.
- `nbcuda/` — CUDA-accelerated neighbor-based routines:
  - `diagonal.py`, `fur.py`, `furx.cu`, `gates.py`, `qaoa_fur.py`, `qaoa_simulator.py`, `utils.py` — GPU paths for diagonal precomputation and XY-ring gate execution.
- `mpi_nbcuda/` — MPI-enabled CUDA variant for distributed runs:
  - `compute_costs.py`, `fur.py`, `qaoa_fur.py`, `qaoa_simulator.py`, `utils.py`, `__init__.py`.
- `diagonal_precomputation/gpu_numba.py`, `diagonal_precomputation/numpy_vectorized.py` — Precomputation of diagonal energies with GPU (Numba) and NumPy backends.
- `c/` (legacy and C-only variants):
  - `energy_qaoa_simulator.py`, `qaoa_simulator.py` (Old_1/Old_2/Old_3) — Historical implementations.
  - `csim/src/diagonal.c`, `diagonal.h` — C routines to build diagonal Hamiltonian terms.
  - `csim/src/fur.c`, `fur.h`, `qaoa_fur.c`, `qaoa_fur.h` — C implementations of fast unitary ring (FUR) simulators.
  - `csim/src/build.sh`, `csim/src/Makefile`, `csim/lib.py`, `csim/wrapper.py`, `csim/libpath.py` — Build tooling, Python bindings, and loader for C libraries.

### Core Energy and Objective Modules
- `energy_qaoa_objective.py` — Bridges diagonal energies with QAOA objective computation (expectation/overlap).
- `energy_utils.py` — Utilities for energy aggregation, indexing, and diagonal construction.
- `energy_labs.py`, `labs.py` — LABS (Low-Autocorrelation Binary Sequences) problem encoding and energy evaluation.
- `energy_maxcut.py`, `maxcut.py` — MaxCut encoding and energy evaluation for graphs.
- `dicke_state_utils.py` — Construction and manipulation of Dicke states for fixed-Hamming-weight subspaces.
- `sk.py`, `energy_qaoa_objective.py` — Sherrington–Kirkpatrick model encodings and QAOA objective linkage.
- `matrix.py` — Helpers for matrix creation and transformations used across simulators.
- `generator.py` — Problem generator and parameter sampling utilities.
- `utils.py` — General-purpose utilities (indexing, bitstring conversions, caching, precompute helpers).

### QAOA Circuits and Objectives
- `qaoa_circuit.py` — Shared circuit construction logic.
- `qaoa_circuit_labs.py` — LABS-specific QAOA circuit assembly.
- `qaoa_circuit_maxcut.py` — MaxCut circuit assembly including cost and mixer layers.
- `qaoa_circuit_portfolio.py` — Portfolio-optimization circuit assembly (Dicke initialization, XY/Trotter mixers).
- `qaoa_circuit_sk.py` — SK-model circuit assembly.
- `qaoa_objective.py` — General objective builders for QAOA (expectation, overlap, parameterization).
- `qaoa_objective_labs.py` — LABS-specific objective functions and precomputation flows.
- `qaoa_objective_maxcut.py` — MaxCut-specific objectives.
- `qaoa_objective_portfolio.py` — Portfolio-specific objective computation and precomputation integration.
- `qaoa_objective_sk.py` — SK-specific objective computation.

### Integration & Diagnostics
- `compile_and_run.py` — Orchestrates building native components and running integration tests for simulators.
- `diagnostic_tool.py` — Runtime diagnostics for performance and correctness across simulator backends.
- `interconnect.py` — Lightweight component registry and routing layer, enabling modular algorithm pieces to interact (used by agents or external drivers).
- `interconnect_diagnostic.py` — Diagnostics for the interconnect layer.
- `integration_examples.py` — Examples showing end-to-end integration across different backends and objectives.
- `parameter_utils.py` — Parameter initialization, scaling, and conversions (e.g., ring mixers vs. RX mixers).
- `perf_utils.py` — Performance measurement and profiling helpers.
- `portfolio_optimization.py` — Mean–Variance portfolio problem creation, data fetching, objective formulation, and brute-force baseline.
- `yahoo.py` — Data provider for Yahoo Finance via `yfinance` with caching and date/ticker handling.
- `quantum_integration_config.json` — Configuration for integration runs and reporting.
- `quantum_integration_report.json` — Output report produced by integration routines.

### Tests
Path (in earlier distributions): `QOKit_enhanced_MLMVN-main/tests/`
- `test_classical_utils.py` — Validates classical helper functions.
- `test_generate_lp.py` — Ensures LP/MIP generation is correct (constraints/objectives).
- `test_matrix.py` — Checks matrix helpers and transformations.
- `test_maxcut.py`, `test_labs.py`, `test_sk.py` (or `test_qaoa_objective_sk.py`) — Problem-specific correctness for encodings and objectives.
- `test_portfolio_optimization.py` — Validates portfolio data fetching, energy computation, and circuit/objective assembly.
- `test_qaoa_objective_*` — Objective-specific test suites for LABS, MaxCut, SK.
- `test_simulator_build.py`, `test_fast_simulators_labs.py`, `test_simulators_rxy.py` — Build/run validation for compiled/GPU simulators and XY-ring gates.
- `test_utils.py` — Utility correctness tests.
- `test_examples_in_the_paper.py` — Reproduces results presented in the associated paper for verification.
- `test_qaoa_qiskit.py` — Qiskit integration tests for circuit execution paths.
- `sample_from_weighted_Shaydulin_Lotshaw_2022.json` — Sample configuration for weighted tests from literature.

## MLMVN (Neural Components)
Path: `QOKit_enhanced_MLMVN-main/MLMVN/`
- `complex_mvn.py` — Complex-valued multi-layer neural network components (MVN), used for hybrid classical–quantum pipelines or parameter prediction.
- `mlmvn_network.py` — MVN network definition, training loops, and evaluation utilities.

## How Pieces Fit Together
- Data → Portfolio:
  - `yahoo.py` fetches prices; `portfolio_optimization.py` constructs returns, μ, Σ, and energy function `q xᵀΣx − μᵀx` with K-hot (budget) constraint.
  - `qaoa_circuit_portfolio.py` builds Dicke-initialized circuits with XY or RX mixers.
  - `qaoa_objective_portfolio.py` computes expectation or overlap using diagonal energy precomputation from `energy_qaoa_objective.py`.
- Simulators:
  - Python/GPU/C backends under `fur/` provide fast diagonal phase updates and XY-ring application. `compile_and_run.py` builds C libraries when needed.
- Other problems:
  - LABS, MaxCut, SK follow parallel structure: encode energies, assemble circuits, construct objectives, and validate via tests/notebooks.
- Parameter management:
  - `parameter_utils.py` provides scaling and initialization across mixers/ansätze; assets include lookup tables for specific graphs or models.
- Integration:
  - `interconnect.py` enables decoupled components (e.g., agents, optimizers) to route requests to core modules, used in advanced workflows.

## Notes
- Some subtrees (e.g., `fur/mpi_nbcuda`) depend on optional system/GPU libraries; build steps are scripted under `fur/c/csim/src`.
- Notebooks under `QOKit/examples` demonstrate end-to-end usage for each problem family and provide reference results.

