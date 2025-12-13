from collections.abc import Sequence
import numpy as np
from numba import jit, njit, prange
from numba.types import float64, int64, boolean


@njit(float64(float64[:], float64[:], boolean), cache=True, fastmath=True, parallel=False)
def _compute_expectation_core(probabilities, costs, is_max):
    """Optimized core computation for expectation value."""
    if is_max:
        result = 0.0
        for i in prange(len(probabilities)):
            result += (-costs[i]) * probabilities[i]
        return result
    else:
        result = 0.0
        for i in prange(len(probabilities)):
            result += costs[i] * probabilities[i]
        return result


@njit(float64[:](float64[:]), cache=True, fastmath=True, parallel=True)
def _negate_costs_vectorized(costs):
    """Ultra-fast vectorized cost negation for max optimization."""
    result = np.empty_like(costs)
    for i in prange(len(costs)):
        result[i] = -costs[i]
    return result


@njit(int64[:](float64[:]), cache=True, fastmath=True, parallel=True)
def _find_min_indices_vectorized(costs):
    """Optimized minimum value indices finder."""
    minval = costs[0]
    # Find minimum value
    for i in prange(1, len(costs)):
        if costs[i] < minval:
            minval = costs[i]
    
    # Count indices with minimum value
    count = 0
    for i in prange(len(costs)):
        if costs[i] == minval:
            count += 1
    
    # Collect indices
    indices = np.empty(count, dtype=np.int64)
    idx = 0
    for i in prange(len(costs)):
        if costs[i] == minval:
            indices[idx] = i
            idx += 1
    
    return indices


@njit(float64(float64[:], int64[:]), cache=True, fastmath=True, parallel=True)
def _sum_probabilities_at_indices(probabilities, indices):
    """Ultra-fast probability summation at specified indices."""
    result = 0.0
    for i in prange(len(indices)):
        result += probabilities[indices[i]]
    return result


@njit(float64(float64[:], float64[:], int64[:], boolean), cache=True, fastmath=True, parallel=False)
def _compute_overlap_with_indices(probabilities, costs, indices, is_max):
    """Optimized overlap computation when indices are provided."""
    return _sum_probabilities_at_indices(probabilities, indices)


@njit(float64(float64[:], float64[:], boolean), cache=True, fastmath=True, parallel=False)
def _compute_overlap_without_indices(probabilities, costs, is_max):
    """Optimized overlap computation when indices need to be computed."""
    if is_max:
        negated_costs = _negate_costs_vectorized(costs)
        indices = _find_min_indices_vectorized(negated_costs)
    else:
        indices = _find_min_indices_vectorized(costs)
    
    return _sum_probabilities_at_indices(probabilities, indices)


def get_expectation(probabilities: np.ndarray, costs: np.ndarray | None = None, optimization_type: str = "min", **kwargs) -> float:
    """Compute the expectation value of the cost Hamiltonian with maximum optimization.

    Parameters
    ----------
    probabilities: Array of probabilities for each state.
    costs: Diagonal of the cost Hamiltonian.
    optimization_type: Optimization type ('min' or 'max').

    Returns
    -------
    float: Expectation value of the cost Hamiltonian.
    """
    if costs is None:
        raise ValueError("Costs must be provided when computing expectation.")
    
    # Convert to contiguous float64 arrays for maximum performance
    prob_array = np.ascontiguousarray(probabilities, dtype=np.float64)
    cost_array = np.ascontiguousarray(costs, dtype=np.float64)
    
    is_max = optimization_type == "max"
    
    return _compute_expectation_core(prob_array, cost_array, is_max)


def get_overlap(
    probabilities: np.ndarray,
    costs: np.ndarray | None = None,
    indices: np.ndarray | Sequence[int] | None = None,
    optimization_type: str = "min",
    **kwargs
) -> float:
    """
    Compute the overlap between the statevector and the ground state with maximum optimization.

    Parameters
    ----------
    probabilities: Array of probabilities for each state.
    costs: (optional) Diagonal of the cost Hamiltonian.
    indices: (optional) Indices of the ground state in the statevector.
    optimization_type: Optimization type ('min' or 'max').

    Returns
    -------
    float: Overlap value.
    """
    # Convert probabilities to optimized format
    prob_array = np.ascontiguousarray(probabilities, dtype=np.float64)
    is_max = optimization_type == "max"
    
    if indices is not None:
        # Fast path when indices are provided
        if isinstance(indices, (list, tuple)):
            indices_array = np.ascontiguousarray(indices, dtype=np.int64)
        else:
            indices_array = np.ascontiguousarray(indices.flatten(), dtype=np.int64)
        
        return _sum_probabilities_at_indices(prob_array, indices_array)
    
    else:
        # Compute indices from costs
        if costs is None:
            raise ValueError("Costs must be provided when indices are not specified.")
        
        cost_array = np.ascontiguousarray(costs, dtype=np.float64)
        return _compute_overlap_without_indices(prob_array, cost_array, is_max)


# Precompiled utility functions for advanced use cases
@njit(float64[:](float64[:], float64), cache=True, fastmath=True, parallel=True)
def add_scalar_vectorized(array, scalar):
    """Ultra-fast scalar addition to array."""
    result = np.empty_like(array)
    for i in prange(len(array)):
        result[i] = array[i] + scalar
    return result


@njit(float64[:](float64[:], float64[:]), cache=True, fastmath=True, parallel=True)
def multiply_arrays_vectorized(arr1, arr2):
    """Ultra-fast element-wise array multiplication."""
    result = np.empty_like(arr1)
    for i in prange(len(arr1)):
        result[i] = arr1[i] * arr2[i]
    return result


@njit(float64(float64[:]), cache=True, fastmath=True, parallel=True)
def sum_vectorized(array):
    """Ultra-fast parallel array summation."""
    result = 0.0
    for i in prange(len(array)):
        result += array[i]
    return result


@njit(float64[:](float64[:]), cache=True, fastmath=True, parallel=True)
def abs_vectorized(array):
    """Ultra-fast absolute value computation."""
    result = np.empty_like(array)
    for i in prange(len(array)):
        result[i] = abs(array[i])
    return result


@njit(int64(float64[:], float64), cache=True, fastmath=True, parallel=False)
def count_equal_values(array, value):
    """Ultra-fast counting of equal values."""
    count = 0
    for i in range(len(array)):
        if array[i] == value:
            count += 1
    return count


# Advanced memory-efficient operations for large-scale simulations
@njit(float64[:](float64[:], int64[:]), cache=True, fastmath=True, parallel=True)
def gather_vectorized(array, indices):
    """Ultra-fast gathering of array elements at specified indices."""
    result = np.empty(len(indices), dtype=np.float64)
    for i in prange(len(indices)):
        result[i] = array[indices[i]]
    return result


@njit(cache=True, fastmath=True, parallel=True)
def scatter_add_vectorized(target, indices, values):
    """Ultra-fast scatter-add operation."""
    for i in prange(len(indices)):
        target[indices[i]] += values[i]


# Memory layout optimization utilities
def ensure_optimal_layout(array):
    """Ensure array has optimal memory layout for maximum performance."""
    return np.ascontiguousarray(array, dtype=np.float64)


def prepare_arrays_for_computation(*arrays):
    """Prepare multiple arrays for optimal computation."""
    return tuple(ensure_optimal_layout(arr) for arr in arrays)


# Advanced bitwise operations for quantum state manipulation
@njit(int64(int64), cache=True, fastmath=True)
def popcount_single(x):
    """Ultra-fast population count for single integer."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


@njit(int64[:](int64[:]), cache=True, fastmath=True, parallel=True)
def popcount_vectorized(array):
    """Ultra-fast vectorized population count."""
    result = np.empty_like(array)
    for i in prange(len(array)):
        result[i] = popcount_single(array[i])
    return result