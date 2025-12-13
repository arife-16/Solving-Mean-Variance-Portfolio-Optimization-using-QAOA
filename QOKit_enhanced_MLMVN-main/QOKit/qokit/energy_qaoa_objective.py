import numpy as np
from numba import jit, prange, types, uint64, float64, int64
from numba.typed import List as NumbaList
import numba


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def _dot_product_parallel_chunks(arr1: np.ndarray, arr2: np.ndarray, chunk_size: int64) -> float64:
    """Оптимизированное векторное произведение с chunking для кэш-локальности."""
    n = arr1.shape[0]
    result = 0.0
    
    # Обработка полных чанков
    full_chunks = n // chunk_size
    for chunk_idx in range(full_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        chunk_sum = 0.0
        
        # Векторизованная обработка чанка с loop unrolling
        for i in range(start_idx, end_idx, 4):
            if i + 3 < end_idx:
                chunk_sum += (arr1[i] * arr2[i] + 
                             arr1[i+1] * arr2[i+1] + 
                             arr1[i+2] * arr2[i+2] + 
                             arr1[i+3] * arr2[i+3])
            else:
                for j in range(i, min(i+4, end_idx)):
                    chunk_sum += arr1[j] * arr2[j]
        
        result += chunk_sum
    
    # Обработка остатка
    remainder_start = full_chunks * chunk_size
    for i in range(remainder_start, n):
        result += arr1[i] * arr2[i]
    
    return result


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _dot_product_parallel_reduce(arr1: np.ndarray, arr2: np.ndarray) -> float64:
    """Параллельное векторное произведение с reduction."""
    n = arr1.shape[0]
    result = 0.0
    
    for i in prange(n):
        result += arr1[i] * arr2[i]
    
    return result


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def _select_optimal_dot_strategy(size: int64) -> int64:
    """Автоматический выбор оптимальной стратегии вычисления dot product."""
    if size < 1000:
        return 0  # Sequential
    elif size < 100000:
        return 1  # Chunked
    else:
        return 2  # Parallel


@jit(nopython=True, cache=True, fastmath=True)
def _compute_dot_product_optimized(arr1: np.ndarray, arr2: np.ndarray) -> float64:
    """Адаптивное векторное произведение с выбором оптимальной стратегии."""
    n = arr1.shape[0]
    strategy = _select_optimal_dot_strategy(n)
    
    if strategy == 0:
        # Последовательное с loop unrolling
        result = 0.0
        i = 0
        # Обработка блоками по 8 элементов
        while i + 7 < n:
            result += (arr1[i] * arr2[i] + arr1[i+1] * arr2[i+1] + 
                      arr1[i+2] * arr2[i+2] + arr1[i+3] * arr2[i+3] +
                      arr1[i+4] * arr2[i+4] + arr1[i+5] * arr2[i+5] + 
                      arr1[i+6] * arr2[i+6] + arr1[i+7] * arr2[i+7])
            i += 8
        # Обработка остатка
        while i < n:
            result += arr1[i] * arr2[i]
            i += 1
        return result
    elif strategy == 1:
        # Chunked для средних размеров
        chunk_size = int64(max(64, min(1024, n // 16)))
        return _dot_product_parallel_chunks(arr1, arr2, chunk_size)
    else:
        # Параллельное для больших размеров
        return _dot_product_parallel_reduce(arr1, arr2)


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _compute_overlap_parallel(probabilities: np.ndarray, bitstring_indices: np.ndarray) -> float64:
    """Параллельное вычисление overlap с оптимизацией доступа к памяти."""
    n_indices = bitstring_indices.shape[0]
    overlap = 0.0
    
    # Параллельная редукция для больших массивов индексов
    if n_indices > 100:
        for i in prange(n_indices):
            idx = bitstring_indices[i]
            overlap += probabilities[idx]
    else:
        # Последовательное для малых массивов с prefetching
        for i in range(n_indices):
            idx = bitstring_indices[i]
            overlap += probabilities[idx]
    
    return overlap


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def _validate_inputs_fast(probabilities_size: int64, precomputed_size: int64, 
                         bitstring_size: int64, has_bitstring: bool) -> bool:
    """Быстрая валидация входных данных."""
    if probabilities_size != precomputed_size:
        return False
    if has_bitstring and bitstring_size > probabilities_size:
        return False
    return True


@jit(nopython=True, cache=True, fastmath=True)
def _compute_expectation_min(precomputed_objectives: np.ndarray, probabilities: np.ndarray) -> float64:
    """Специализированная функция для expectation + min."""
    return _compute_dot_product_optimized(precomputed_objectives, probabilities)


@jit(nopython=True, cache=True, fastmath=True)
def _compute_expectation_max(precomputed_objectives: np.ndarray, probabilities: np.ndarray) -> float64:
    """Специализированная функция для expectation + max."""
    dot_result = _compute_dot_product_optimized(precomputed_objectives, probabilities)
    return -dot_result


@jit(nopython=True, cache=True, fastmath=True)
def _compute_overlap_objective(probabilities: np.ndarray, bitstring_indices: np.ndarray) -> float64:
    """Специализированная функция для overlap."""
    overlap = _compute_overlap_parallel(probabilities, bitstring_indices)
    return 1.0 - overlap


# Функции уже скомпилированы декораторами выше, используем их напрямую
compute_expectation_min_compiled = _compute_expectation_min
compute_expectation_max_compiled = _compute_expectation_max
compute_overlap_compiled = _compute_overlap_objective


def compute_objective_from_probabilities(
    probabilities: np.ndarray,
    objective: str,
    precomputed_objectives: np.ndarray,
    bitstring_loc: np.ndarray | None = None,
    optimization_type: str = "min"
) -> float:
    """Compute the objective value from probabilities with maximum optimization.

    Args:
        probabilities: Array of probabilities for each state.
        objective: Type of objective ('expectation' or 'overlap').
        precomputed_objectives: Precomputed objective values for each state.
        bitstring_loc: Indices of optimal bitstrings (for overlap objective).
        optimization_type: Optimization type ('min' or 'max').

    Returns:
        float: Objective value.
    """
    # Предварительная валидация с быстрой проверкой
    if not _validate_inputs_fast(
        int64(len(probabilities)), 
        int64(len(precomputed_objectives)),
        int64(len(bitstring_loc)) if bitstring_loc is not None else int64(0),
        bitstring_loc is not None
    ):
        if len(probabilities) != len(precomputed_objectives):
            raise ValueError("Probabilities and precomputed_objectives must have the same length")
    
    # Конвертация в нужные типы для максимальной производительности
    prob_array = probabilities.astype(np.float64, copy=False)
    precomp_array = precomputed_objectives.astype(np.float64, copy=False)
    
    if objective == "expectation":
        if optimization_type == "max":
            return float(compute_expectation_max_compiled(precomp_array, prob_array))
        else:  # optimization_type == "min"
            return float(compute_expectation_min_compiled(precomp_array, prob_array))
    
    elif objective == "overlap":
        if bitstring_loc is None:
            raise ValueError("bitstring_loc cannot be None for overlap objective")
        
        # Конвертация индексов в int64 для максимальной производительности
        bitstring_indices = bitstring_loc.astype(np.int64, copy=False)
        return float(compute_overlap_compiled(prob_array, bitstring_indices))
    
    else:
        raise ValueError(f"Unknown objective: {objective}, allowed ['expectation', 'overlap']")


# Дополнительные функции для специализированных случаев использования
@jit(nopython=True, cache=True, fastmath=True)
def compute_expectation_batch(probabilities_batch: np.ndarray, 
                             precomputed_objectives: np.ndarray,
                             optimization_type_code: int64) -> np.ndarray:
    """Батчевое вычисление expectation для множественных распределений вероятностей.
    
    Args:
        probabilities_batch: 2D массив (n_batch, n_states)
        precomputed_objectives: 1D массив (n_states,)
        optimization_type_code: 0 для min, 1 для max
        
    Returns:
        1D массив результатов (n_batch,)
    """
    n_batch = probabilities_batch.shape[0]
    results = np.empty(n_batch, dtype=np.float64)
    
    if optimization_type_code == 0:  # min
        for i in range(n_batch):
            results[i] = _compute_dot_product_optimized(precomputed_objectives, probabilities_batch[i])
    else:  # max
        for i in range(n_batch):
            dot_result = _compute_dot_product_optimized(precomputed_objectives, probabilities_batch[i])
            results[i] = -dot_result
    
    return results


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def compute_expectation_batch_parallel(probabilities_batch: np.ndarray, 
                                     precomputed_objectives: np.ndarray,
                                     optimization_type_code: int64) -> np.ndarray:
    """Параллельное батчевое вычисление expectation."""
    n_batch = probabilities_batch.shape[0]
    results = np.empty(n_batch, dtype=np.float64)
    
    if optimization_type_code == 0:  # min
        for i in prange(n_batch):
            results[i] = _compute_dot_product_optimized(precomputed_objectives, probabilities_batch[i])
    else:  # max
        for i in prange(n_batch):
            dot_result = _compute_dot_product_optimized(precomputed_objectives, probabilities_batch[i])
            results[i] = -dot_result
    
    return results