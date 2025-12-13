from collections.abc import Sequence
import numpy as np
from numba import jit, njit, prange, uint64, float64, int64, boolean
from numba.types import Array, Tuple as NumbaT
from typing import Tuple


# Углубленные типы данных для максимальной производительности
float64_1d = float64[:]
int64_1d = int64[:]
boolean_scalar = boolean


@njit(float64(float64_1d, float64_1d, boolean_scalar), 
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'result': float64, 'i': int64})
def _compute_expectation_core_advanced(probabilities, costs, is_max):
    """Максимально оптимизированное ядро вычисления expectation value с углубленной JIT-компиляцией."""
    result = float64(0.0)
    
    if is_max:
        for i in prange(len(probabilities)):
            result += probabilities[i] * (-costs[i])
    else:
        for i in prange(len(probabilities)):
            result += probabilities[i] * costs[i]
    
    return result


@njit(float64_1d(float64_1d), 
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def _negate_costs_ultra_optimized(costs):
    """Ультра-быстрое векторизованное отрицание стоимостей с углубленной оптимизацией."""
    n = len(costs)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = -costs[i]
    
    return result


@njit(NumbaT((float64, int64_1d))(float64_1d),
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'minval': float64, 'i': int64, 'count': int64, 'idx': int64})
def _find_min_indices_ultra_optimized(costs):
    """Максимально оптимизированный поиск индексов минимальных значений."""
    n = len(costs)
    minval = costs[0]
    
    # Поиск минимального значения с развернутым циклом для оптимизации
    for i in range(1, n):
        if costs[i] < minval:
            minval = costs[i]
    
    # Подсчет количества минимальных элементов
    count = int64(0)
    for i in range(n):
        if costs[i] == minval:
            count += int64(1)
    
    # Сбор индексов с предварительно выделенной памятью
    indices = np.empty(count, dtype=np.int64)
    idx = int64(0)
    for i in range(n):
        if costs[i] == minval:
            indices[idx] = i
            idx += int64(1)
    
    return minval, indices


@njit(float64(float64_1d, int64_1d),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'result': float64, 'i': int64})
def _sum_probabilities_at_indices_advanced(probabilities, indices):
    """Ультра-быстрое суммирование вероятностей с углубленной оптимизацией."""
    result = float64(0.0)
    
    for i in prange(len(indices)):
        result += probabilities[indices[i]]
    
    return result


@njit(float64(float64_1d, float64_1d, int64_1d, boolean_scalar),
      cache=True, fastmath=True, parallel=False, nogil=True)
def _compute_overlap_with_indices_advanced(probabilities, costs, indices, is_max):
    """Оптимизированное вычисление overlap с предоставленными индексами."""
    return _sum_probabilities_at_indices_advanced(probabilities, indices)


@njit(float64(float64_1d, float64_1d, boolean_scalar),
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'minval': float64, 'indices': int64_1d})
def _compute_overlap_without_indices_advanced(probabilities, costs, is_max):
    """Максимально оптимизированное вычисление overlap без предоставленных индексов."""
    if is_max:
        negated_costs = _negate_costs_ultra_optimized(costs)
        minval, indices = _find_min_indices_ultra_optimized(negated_costs)
    else:
        minval, indices = _find_min_indices_ultra_optimized(costs)
    
    return _sum_probabilities_at_indices_advanced(probabilities, indices)


# Дополнительная углубленная оптимизация памяти
@njit(float64_1d(float64_1d), 
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'i': int64})
def _ensure_contiguous_float64(array):
    """Обеспечение оптимального размещения в памяти для максимальной производительности."""
    n = len(array)
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        result[i] = array[i]
    
    return result


@njit(int64_1d(int64_1d),
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'i': int64})
def _ensure_contiguous_int64(array):
    """Обеспечение оптимального размещения в памяти для индексов."""
    n = len(array)
    result = np.empty(n, dtype=np.int64)
    
    for i in range(n):
        result[i] = array[i]
    
    return result


def get_expectation(probabilities: np.ndarray, costs: np.ndarray | None = None, optimization_type: str = "min", **kwargs) -> float:
    """Максимально оптимизированное вычисление expectation value с углубленной JIT-компиляцией.

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
    
    # Максимальная оптимизация памяти для углубленной производительности
    prob_array = np.ascontiguousarray(probabilities, dtype=np.float64)
    cost_array = np.ascontiguousarray(costs, dtype=np.float64)
    
    # Предварительная проверка размеров для оптимизации
    if len(prob_array) != len(cost_array):
        raise ValueError("Probabilities and costs arrays must have the same length.")
    
    is_max = optimization_type == "max"
    
    return _compute_expectation_core_advanced(prob_array, cost_array, is_max)


def get_overlap(
    probabilities: np.ndarray,
    costs: np.ndarray | None = None,
    indices: np.ndarray | Sequence[int] | None = None,
    optimization_type: str = "min",
    **kwargs
) -> float:
    """
    Максимально оптимизированное вычисление overlap с углубленными методами JIT-компиляции.

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
    # Углубленная оптимизация памяти
    prob_array = np.ascontiguousarray(probabilities, dtype=np.float64)
    is_max = optimization_type == "max"
    
    if indices is not None:
        # Максимально быстрый путь с предоставленными индексами
        if isinstance(indices, (list, tuple)):
            indices_array = np.ascontiguousarray(indices, dtype=np.int64)
        else:
            indices_array = np.ascontiguousarray(indices.flatten(), dtype=np.int64)
        
        return _sum_probabilities_at_indices_advanced(prob_array, indices_array)
    
    else:
        # Вычисление индексов из costs с углубленной оптимизацией
        if costs is None:
            raise ValueError("Costs must be provided when indices are not specified.")
        
        cost_array = np.ascontiguousarray(costs, dtype=np.float64)
        return _compute_overlap_without_indices_advanced(prob_array, cost_array, is_max)


# Углубленные утилитарные функции с максимальной оптимизацией
@njit(float64_1d(float64_1d, float64),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def add_scalar_vectorized_advanced(array, scalar):
    """Ультра-быстрое добавление скаляра к массиву с углубленной оптимизацией."""
    n = len(array)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = array[i] + scalar
    
    return result


@njit(float64_1d(float64_1d, float64_1d),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def multiply_arrays_vectorized_advanced(arr1, arr2):
    """Ультра-быстрое поэлементное умножение массивов с углубленной оптимизацией."""
    n = len(arr1)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = arr1[i] * arr2[i]
    
    return result


@njit(float64(float64_1d),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'result': float64, 'i': int64})
def sum_vectorized_advanced(array):
    """Ультра-быстрое параллельное суммирование массива с углубленной оптимизацией."""
    result = float64(0.0)
    
    for i in prange(len(array)):
        result += array[i]
    
    return result


@njit(float64_1d(float64_1d),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def abs_vectorized_advanced(array):
    """Ультра-быстрое вычисление абсолютного значения с углубленной оптимизацией."""
    n = len(array)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        val = array[i]
        result[i] = val if val >= 0.0 else -val
    
    return result


@njit(int64(float64_1d, float64),
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'count': int64, 'i': int64})
def count_equal_values_advanced(array, value):
    """Ультра-быстрый подсчет равных значений с углубленной оптимизацией."""
    count = int64(0)
    
    for i in range(len(array)):
        if array[i] == value:
            count += int64(1)
    
    return count


# Углубленные операции с памятью для крупномасштабных симуляций
@njit(float64_1d(float64_1d, int64_1d),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def gather_vectorized_advanced(array, indices):
    """Ультра-быстрое сбор элементов массива по указанным индексам с углубленной оптимизацией."""
    n = len(indices)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        result[i] = array[indices[i]]
    
    return result


@njit(cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def scatter_add_vectorized_advanced(target, indices, values):
    """Ультра-быстрая операция scatter-add с углубленной оптимизацией."""
    for i in prange(len(indices)):
        target[indices[i]] += values[i]


# Углубленные битовые операции для квантовых состояний
@njit(uint64(uint64),
      cache=True, fastmath=True, nogil=True,
      locals={'count': uint64})
def popcount_single_advanced(x):
    """Ультра-быстрый подсчет битов для одного целого числа с углубленной оптимизацией."""
    count = uint64(0)
    
    # Оптимизированный алгоритм подсчета битов
    while x:
        count += uint64(1)
        x &= x - uint64(1)  # Убирает младший установленный бит
    
    return count


@njit(uint64[::1](uint64[::1]),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def popcount_vectorized_advanced(array):
    """Ультра-быстрый векторизованный подсчет битов с углубленной оптимизацией."""
    n = len(array)
    result = np.empty(n, dtype=np.uint64)
    
    for i in prange(n):
        result[i] = popcount_single_advanced(array[i])
    
    return result


# Дополнительные углубленные операции для максимальной производительности
@njit(boolean(float64_1d, float64_1d, float64),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def arrays_equal_within_tolerance_advanced(arr1, arr2, tolerance):
    """Ультра-быстрая проверка равенства массивов с допуском."""
    if len(arr1) != len(arr2):
        return False
    
    for i in prange(len(arr1)):
        diff = arr1[i] - arr2[i]
        if diff < 0.0:
            diff = -diff
        if diff > tolerance:
            return False
    
    return True


@njit(float64_1d(float64_1d, float64, float64),
      cache=True, fastmath=True, parallel=True, nogil=True,
      locals={'i': int64})
def clip_vectorized_advanced(array, min_val, max_val):
    """Ультра-быстрое ограничение значений массива с углубленной оптимизацией."""
    n = len(array)
    result = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        val = array[i]
        if val < min_val:
            result[i] = min_val
        elif val > max_val:
            result[i] = max_val
        else:
            result[i] = val
    
    return result


@njit(NumbaT((float64, float64))(float64_1d),
      cache=True, fastmath=True, parallel=False, nogil=True,
      locals={'min_val': float64, 'max_val': float64, 'i': int64})
def find_min_max_advanced(array):
    """Ультра-быстрый поиск минимума и максимума с углубленной оптимизацией."""
    min_val = array[0]
    max_val = array[0]
    
    for i in range(1, len(array)):
        val = array[i]
        if val < min_val:
            min_val = val
        elif val > max_val:
            max_val = val
    
    return min_val, max_val


# Функции для оптимального размещения в памяти
def ensure_optimal_layout_advanced(array):
    """Обеспечение оптимального размещения массива в памяти для максимальной производительности."""
    return np.ascontiguousarray(array, dtype=np.float64)


def prepare_arrays_for_computation_advanced(*arrays):
    """Подготовка нескольких массивов для оптимального вычисления с углубленной оптимизацией."""
    result = []
    for arr in arrays:
        if arr is not None:
            result.append(ensure_optimal_layout_advanced(arr))
        else:
            result.append(None)
    return tuple(result)