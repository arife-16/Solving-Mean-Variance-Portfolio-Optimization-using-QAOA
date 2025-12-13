from collections.abc import Sequence
import numpy as np
from numba import jit, prange, float64, uint64, boolean
from numba.types import Array, Optional


@jit(nopython=True, cache=True, inline='always')
def _compute_abs_squared_vectorized(result_real: np.ndarray, result_imag: np.ndarray) -> np.ndarray:
    """Векторизованное вычисление |z|^2 для комплексных чисел."""
    n = len(result_real)
    probabilities = np.empty(n, dtype=float64)
    
    for i in range(n):
        probabilities[i] = result_real[i] * result_real[i] + result_imag[i] * result_imag[i]
    
    return probabilities


@jit(nopython=True, cache=True, inline='always')
def _compute_abs_squared_real(result: np.ndarray) -> np.ndarray:
    """Оптимизированное вычисление квадрата модуля для вещественных чисел."""
    n = len(result)
    probabilities = np.empty(n, dtype=float64)
    
    for i in range(n):
        val = result[i]
        probabilities[i] = val * val
    
    return probabilities


@jit(nopython=True, parallel=True, cache=True)
def _compute_abs_squared_parallel(result: np.ndarray) -> np.ndarray:
    """Параллельное вычисление квадрата модуля."""
    n = len(result)
    probabilities = np.empty(n, dtype=float64)
    
    for i in prange(n):
        val = result[i]
        probabilities[i] = val * val
    
    return probabilities


@jit(nopython=True, cache=True, inline='always')
def _dot_product_optimized(costs: np.ndarray, probabilities: np.ndarray) -> float64:
    """Высокооптимизированное скалярное произведение с развертыванием цикла."""
    n = len(costs)
    result = float64(0.0)
    
    # Развертывание цикла для векторизации на аппаратном уровне
    i = 0
    while i + 4 <= n:
        result += (costs[i] * probabilities[i] + 
                  costs[i+1] * probabilities[i+1] + 
                  costs[i+2] * probabilities[i+2] + 
                  costs[i+3] * probabilities[i+3])
        i += 4
    
    # Обработка оставшихся элементов
    while i < n:
        result += costs[i] * probabilities[i]
        i += 1
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _dot_product_parallel(costs: np.ndarray, probabilities: np.ndarray) -> float64:
    """Параллельное скалярное произведение для больших массивов."""
    n = len(costs)
    
    # Используем параллельную редукцию
    partial_sums = np.zeros(1, dtype=float64)
    
    for i in prange(n):
        # Атомарное добавление не нужно для float64 сумм в этом контексте
        # так как numba обеспечивает корректную редукцию
        partial_sums[0] += costs[i] * probabilities[i]
    
    return partial_sums[0]


@jit(nopython=True, cache=True)
def _find_extremum_indices_sequential(costs: np.ndarray, find_max: boolean) -> np.ndarray:
    """Последовательный поиск индексов экстремальных значений."""
    n = len(costs)
    if n == 0:
        return np.empty(0, dtype=uint64)
    
    extremum_val = costs[0]
    
    # Первый проход: находим экстремальное значение
    for i in range(1, n):
        if find_max:
            if costs[i] > extremum_val:
                extremum_val = costs[i]
        else:
            if costs[i] < extremum_val:
                extremum_val = costs[i]
    
    # Второй проход: подсчитываем количество экстремальных элементов
    count = 0
    for i in range(n):
        if costs[i] == extremum_val:
            count += 1
    
    # Третий проход: собираем индексы
    indices = np.empty(count, dtype=uint64)
    idx = 0
    for i in range(n):
        if costs[i] == extremum_val:
            indices[idx] = uint64(i)
            idx += 1
    
    return indices


@jit(nopython=True, parallel=True, cache=True)
def _find_extremum_indices_parallel(costs: np.ndarray, find_max: boolean) -> np.ndarray:
    """Параллельный поиск индексов с двухэтапным алгоритмом."""
    n = len(costs)
    if n == 0:
        return np.empty(0, dtype=uint64)
    
    # Параллельный поиск экстремума через редукцию
    extremum_val = costs[0]
    
    if find_max:
        for i in prange(1, n):
            if costs[i] > extremum_val:
                extremum_val = costs[i]
    else:
        for i in prange(1, n):
            if costs[i] < extremum_val:
                extremum_val = costs[i]
    
    # Параллельный подсчет совпадений
    matches = np.zeros(n, dtype=uint64)
    count = 0
    
    for i in prange(n):
        if costs[i] == extremum_val:
            matches[i] = uint64(1)
            count += 1
    
    # Сборка результата
    indices = np.empty(count, dtype=uint64)
    idx = 0
    for i in range(n):
        if matches[i] == 1:
            indices[idx] = uint64(i)
            idx += 1
    
    return indices


@jit(nopython=True, cache=True, inline='always')
def _sum_by_indices_optimized(probabilities: np.ndarray, indices: np.ndarray) -> float64:
    """Оптимизированная сумма по индексам с развертыванием цикла."""
    n = len(indices)
    result = float64(0.0)
    
    # Развертывание цикла
    i = 0
    while i + 4 <= n:
        result += (probabilities[indices[i]] + 
                  probabilities[indices[i+1]] + 
                  probabilities[indices[i+2]] + 
                  probabilities[indices[i+3]])
        i += 4
    
    while i < n:
        result += probabilities[indices[i]]
        i += 1
    
    return result


@jit(nopython=True, parallel=True, cache=True)
def _sum_by_indices_parallel(probabilities: np.ndarray, indices: np.ndarray) -> float64:
    """Параллельная сумма по индексам."""
    n = len(indices)
    partial_sum = float64(0.0)
    
    for i in prange(n):
        partial_sum += probabilities[indices[i]]
    
    return partial_sum


def get_probabilities(result: np.ndarray, **kwargs) -> np.ndarray:
    """Compute probabilities from statevector with deep JIT optimization.
    
    Parameters
    ----------
    result: Statevector array.
    
    Returns
    -------
    np.ndarray: Array of probabilities.
    """
    # Проверка на комплексность входного массива
    if np.iscomplexobj(result):
        # Разделение на действительную и мнимую части для оптимизации
        real_part = np.real(result).astype(np.float64)
        imag_part = np.imag(result).astype(np.float64)
        return _compute_abs_squared_vectorized(real_part, imag_part)
    else:
        # Оптимизация для вещественных чисел
        result_float64 = result.astype(np.float64)
        
        # Автоматический выбор между параллельным и последовательным режимом
        if len(result_float64) > 10000:
            return _compute_abs_squared_parallel(result_float64)
        else:
            return _compute_abs_squared_real(result_float64)


def get_expectation(probabilities: np.ndarray, costs: np.ndarray | None = None, optimization_type: str = "min", **kwargs) -> float:
    """Compute the expectation value of the cost Hamiltonian with JIT acceleration.

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
    
    # Обеспечение типовой совместимости
    costs_float64 = np.asarray(costs, dtype=np.float64)
    probabilities_float64 = np.asarray(probabilities, dtype=np.float64)
    
    # Выбор оптимального алгоритма в зависимости от размера данных
    if len(costs_float64) > 50000:
        dot_result = _dot_product_parallel(costs_float64, probabilities_float64)
    else:
        dot_result = _dot_product_optimized(costs_float64, probabilities_float64)
    
    if optimization_type == "max":
        return -1.0 * dot_result
    return dot_result


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
    probabilities_float64 = np.asarray(probabilities, dtype=np.float64)
    
    if indices is None:
        if costs is None:
            raise ValueError("Costs must be provided when indices are not specified.")
        
        costs_float64 = np.asarray(costs, dtype=np.float64)
        find_max = (optimization_type == "max")
        
        # Выбор алгоритма поиска экстремума
        if len(costs_float64) > 20000:
            indices_array = _find_extremum_indices_parallel(costs_float64, find_max)
        else:
            indices_array = _find_extremum_indices_sequential(costs_float64, find_max)
    else:
        # Преобразование индексов в оптимальный формат
        if isinstance(indices, (list, tuple)):
            indices_array = np.array(indices, dtype=np.uint64)
        else:
            indices_array = np.asarray(indices, dtype=np.uint64)
    
    # Оптимизированное вычисление суммы
    if len(indices_array) > 10000:
        return _sum_by_indices_parallel(probabilities_float64, indices_array)
    else:
        return _sum_by_indices_optimized(probabilities_float64, indices_array)