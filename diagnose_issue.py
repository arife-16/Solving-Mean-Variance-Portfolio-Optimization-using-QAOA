"""
Diagnostic script to identify issues with ADAPT, warm-start, and noise
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_portfolio.portfolio_pipeline import PortfolioPipeline
from quant_portfolio.qaoa_core import warm_start_state, dicke_state
from quant_portfolio.data import generate_synthetic_returns, compute_mu_sigma

print("="*70)
print("DIAGNOSTICS FOR QAMP PROJECT")
print("="*70)

pipe = PortfolioPipeline(seed=42)

# ============================================================================
# TEST 1: Warm-Start Implementation
# ============================================================================
print("\n[TEST 1] Warm-Start Biasing Check")
print("-" * 70)

N, K = 8, 4
problem = pipe._get_problem(N=N, K=K, q=0.5)

# Generate cold start
cold = dicke_state(N, K)
cold_probs = np.abs(cold)**2

# Generate warm start
warm = warm_start_state(problem['means'], problem['cov'], K)
warm_probs = np.abs(warm)**2

# Check if they're different
cold_uniform = np.allclose(cold_probs, 1/70)  # C(8,4) = 70 states
warm_uniform = np.allclose(warm_probs, 1/70)

print(f"Cold start is uniform: {cold_uniform} ✓" if cold_uniform else "Cold start is NOT uniform ✗")
print(f"Warm start is uniform: {warm_uniform}")
print(f"Warm start is biased: {not warm_uniform} {'✓' if not warm_uniform else '✗ PROBLEM!'}")

if not warm_uniform:
    print(f"Warm state entropy: {-np.sum(warm_probs * np.log(warm_probs + 1e-12)):.4f}")
    print(f"Cold state entropy: {-np.sum(cold_probs * np.log(cold_probs + 1e-12)):.4f}")
    print("Lower entropy = more biased (good for warm start)")
else:
    print("⚠️ WARNING: Warm-start is not biasing the initial state!")
    print("This explains why warm-start shows no improvement.")

# ============================================================================
# TEST 2: ADAPT Layer Growth
# ============================================================================
print("\n[TEST 2] ADAPT Layer Growth Test")
print("-" * 70)

for max_layers in [2, 6]:
    print(f"\nTesting with max_layers={max_layers}:")
    
    result = pipe.run_adapt(
        N=8, K=4, q=0.5, 
        max_layers=max_layers,
        mixer='xy',
        warm_start=False,
        formulation='mvo',
        pool='pairs',
        pairs_mode='ring'
    )
    
    print(f"  Layers used: {result['layers']}/{max_layers}")
    print(f"  Energy gap: {result['energy_gap']:.6f}")
    print(f"  2-qubit gates: {result['gate_counts']['two_qubit']}")

print("\n✓ If layers_used == max_layers, ADAPT is maxing out!")
print("  → Need to increase max_layers for ADAPT to work properly")

# ============================================================================
# TEST 3: Ring vs All Pairs
# ============================================================================
print("\n[TEST 3] Operator Pool Comparison")
print("-" * 70)

from quant_portfolio.adapt_qaoa import build_pairs

N = 8
ring_pairs = build_pairs(N, 'ring')
all_pairs = build_pairs(N, 'all')

print(f"Ring topology: {len(ring_pairs)} pairs")
print(f"All pairs: {len(all_pairs)} pairs")
print(f"Ratio: {len(all_pairs) / len(ring_pairs):.1f}x more operators")

print("\n✓ More operators = better adaptive selection")
print("  → Try 'all' pairs for better ADAPT performance")

# ============================================================================
# TEST 4: Noise Application Sanity Check
# ============================================================================
print("\n[TEST 4] Noise Model Sanity Check")
print("-" * 70)

from quant_portfolio.qaoa_core import apply_noise

# Create simple test state
N = 4
test_probs = np.zeros(16)
test_probs[0] = 1.0  # All probability on |0000>

# Apply different noise levels
for noise_p in [0.01, 0.1]:
    noisy_probs = apply_noise(test_probs.copy(), 'bitflip', noise_p, N)
    
    # After bitflip noise, probability should leak to neighbors
    leaked = 1.0 - noisy_probs[0]
    
    print(f"\nNoise p={noise_p}:")
    print(f"  Original prob[0]: {test_probs[0]:.4f}")
    print(f"  Noisy prob[0]: {noisy_probs[0]:.4f}")
    print(f"  Leaked: {leaked:.4f}")
    print(f"  Expected leak: ~{N * noise_p:.4f} (for N bits)")
    
    if abs(leaked - N * noise_p) > 0.05:
        print(f"  ⚠️ WARNING: Leaked amount doesn't match expected!")

# ============================================================================
# TEST 5: Quick Warm vs Cold Comparison
# ============================================================================
print("\n[TEST 5] Quick Warm vs Cold Comparison (N=8)")
print("-" * 70)

for warm in [False, True]:
    result = pipe.run_standard(
        N=8, K=4, q=0.5, p=1,
        mixer='xy',
        warm_start=warm,
        samples=16,
        refine_iters=10
    )
    
    label = "Warm" if warm else "Cold"
    print(f"{label} start: Energy gap = {result['energy_gap']:.6f}")

print("\n✓ If these are identical, warm-start is not working!")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("DIAGNOSTIC SUMMARY")
print("="*70)
print("\nExpected Issues:")
print("1. Warm-start might return uniform state (not biased)")
print("2. ADAPT might max out at 2 layers (needs higher max_layers)")
print("3. Ring topology limits ADAPT operator selection")
print("4. Noise model might be too aggressive")
print("\nNext: Run fixes based on diagnostic results")
print("="*70)