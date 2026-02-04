# Solving Mean-Variance Portfolio Optimization with Quantum Utility

## The Problem: The "Quantum Cliff" in Financial Optimization
Classical financial optimization, particularly Mean-Variance Optimization (MVO), is a staple of modern portfolio theory. However, incorporating discrete constraints (e.g., cardinality constraints like "select exactly $K$ assets") turns this convex problem into an NP-hard Mixed-Integer Quadratic Programming (MIQP) challenge.

While Quantum Approximate Optimization Algorithm (QAOA) offers a promising path, standard implementations hit a **"Quantum Cliff"** when applied to real-world financial data:
1.  **Exploding Circuit Depth**: Standard mixers require deep circuits to enforce constraints, exceeding the coherence time of NISQ hardware.
2.  **Poor Convergence**: On real, correlated market data (unlike synthetic benchmarks), standard QAOA struggles to find the optimal subspace.
3.  **Initialization Bottleneck**: "Cold" starts (uniform superposition) waste precious quantum resources exploring irrelevant regions of the Hilbert space.

## The Solution: A Rigorous, Hardware-Aware Quantum Stack
This repository implements a production-grade quantum optimization pipeline designed to bridge the gap between theoretical QAOA and practical utility. We move beyond "textbook" implementations to address the specific challenges of portfolio optimization.

### Key Innovations

#### 1. True Warm-Start QAOA (WS-QAOA)
Instead of starting from a "cold" uniform superposition, we leverage classical pre-computation to give the quantum algorithm a head start.
*   **Classical Relaxation**: We solve the continuous relaxation of the problem (via `cvxpy` or `scipy` fallback) to obtain optimal continuous weights $w_i \in [0, 1]$.
*   **State Mapping**: These weights are mapped to qubit rotation angles $\theta_i = 2\arcsin(\sqrt{w_i})$, creating a biased initial state $|\phi_0\rangle$.
*   **Rigorous Mixer**: Unlike standard implementations that immediately destroy this bias, we implement the **Modified Warm-Start Mixer** (Egger et al., 2021):
    $$U_M(\beta) = \prod_{i} R_y(\theta_i) R_z(-2\beta) R_y(-\theta_i)$$
    This Hamiltonian preserves $|\phi_0\rangle$ as its ground state, ensuring the quantum evolution refines rather than resets the classical solution.

#### 2. ADAPT-QAOA with Gradient-Based Operator Selection
To tackle the circuit depth problem, we implement **ADAPT-QAOA**, which iteratively grows the ansatz.
*   **Adaptive Growth**: Instead of a fixed depth $p$, the algorithm selects the most impactful operator from a pool (e.g., `XY` pairs) based on energy gradients.
*   **Outcome**: We achieve higher approximation ratios with significantly shallower circuits (typically $p \approx 3-4$ layers) compared to standard QAOA, directly addressing the "Quantum Cliff."

#### 3. Real-World Robustness
*   **CVaR Optimization**: Beyond minimizing expected energy, we support optimizing Conditional Value at Risk (CVaR), focusing on the "tail" of the distribution to find high-quality solutions even when the ground state probability is low.
*   **Connectivity Awareness**: We account for the "Connectivity Tax" of mapping logical qubits to physical linear topologies, providing realistic gate count estimates.
*   **Constraint-Preserving Mixers**: Our XY-Mixer implementation uses block-diagonal matrix exponentiation to strictly preserve the Hamming weight ($K$-hot subspace) with machine precision, unlike penalty-based approaches that waste qubits.

## Technical Architecture

### Core Components
*   **`PortfolioPipeline`**: The central orchestrator that manages data loading, problem formulation, and quantum execution.
*   **`qaoa_core`**: Optimized NumPy/SciPy backend for statevector simulation (scales to $N \approx 26$).
*   **`adapt_qaoa`**: Implementation of the adaptive ansatz growth strategy using gradient criteria.
*   **`mip` & `formulations`**: Robust classical optimization suite (MILP/MIQP) for benchmarking and warm-start relaxation.

### Validated Performance
Our rigorous benchmarking suite confirms:
*   **Subspace Preservation**: The XY-Mixer maintains the $K$-asset constraint with negligible leakage ($< 10^{-15}$).
*   **Warm-Start Advantage**: The "Biased Initialization + XY Mixer" strategy consistently outperforms both Cold Start and constrained Warm-Start dynamics on real market data.
*   **Scalability**: ADAPT-QAOA demonstrates superior scaling on correlated S&P 500 data compared to standard fixed-depth approaches.

## Getting Started

### Installation
```bash
pip install -r requirements.txt
```
*Requires Python 3.9+. Optional: `cvxpy` with commercial solvers (Gurobi, CPLEX) for faster classical benchmarks.*

### Running a Full Experiment
Execute the end-to-end validation script to reproduce our findings on real market data:
```bash
python3 scripts/run_final_validation.py
```
This will run comparative experiments on $N=12$ and $N=16$ assets, testing Cold vs. Warm starts and Standard vs. ADAPT protocols.

### Visualizing Results
Generate the final report plots from the experiment data:
```bash
python3 scripts/plot_final_validation.py
```
Results will be saved to `results/plots_final/`.

## Repository Structure
*   `quant_portfolio/`: Core library code.
*   `scripts/`: Experiment orchestration and plotting scripts.
*   `results/`: Data artifacts and generated plots.
*   `tests/`: Unit tests for critical components.

---
*This project was developed as part of the QAMP mentorship program, focusing on rigorous, hardware-aware quantum algorithm development.*
