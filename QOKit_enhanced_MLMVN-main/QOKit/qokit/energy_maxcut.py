###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
from numba import jit, prange, types, typed, uint64, int64, float64, boolean
from numba.core import config
from numba.extending import overload_method
from numba.core.types import Array, float64 as nb_float64, int64 as nb_int64
import numba as nb
from typing import Optional, Tuple, List
import math

# Конфигурация максимальной оптимизации
config.DISABLE_JIT = False
config.DUMP_ASSEMBLY = False
config.NUMBA_DISABLE_INTEL_SVML = False  # Включаем Intel SVML для математических функций

# Глобальные константы для оптимизации
CACHE_LINE_SIZE = 64
SIMD_WIDTH = 8
PARALLEL_THRESHOLD = 1000
SPARSE_THRESHOLD = 0.1
SYMMETRIC_THRESHOLD = 50

@jit(nopython=True, cache=True, fastmath=True, inline='always', boundscheck=False)
def _ultra_fast_contribution(x_i: float64, x_j: float64, w_ij: float64) -> float64:
    """Максимально оптимизированное вычисление вклада одного элемента."""
    # Прямое вычисление без промежуточных переменных
    return w_ij * x_i * (1.0 - x_j)

@jit(nopython=True, cache=True, fastmath=True, inline='always', boundscheck=False)
def _fused_multiply_add(a: float64, b: float64, c: float64) -> float64:
    """Использование FMA (Fused Multiply-Add) для ускорения арифметических операций."""
    return a * b + c

@jit(nopython=True, cache=True, fastmath=True, inline='always', boundscheck=False)
def _vectorized_dot_product(x: np.ndarray, y: np.ndarray, start: int64, end: int64) -> float64:
    """Векторизованное скалярное произведение для сегмента массива."""
    result = 0.0
    for i in range(start, end):
        result = _fused_multiply_add(x[i], y[i], result)
    return result

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def _maxcut_parallel_tiled(x: np.ndarray, w: np.ndarray, tile_size: int64) -> float64:
    """Параллельное тайловое вычисление для максимального использования кэша."""
    n = x.shape[0]
    total = 0.0
    
    # Разбиение на тайлы для оптимизации кэша
    n_tiles = (n + tile_size - 1) // tile_size
    
    for tile_i in prange(n_tiles):
        local_sum = 0.0
        start_i = tile_i * tile_size
        end_i = min(start_i + tile_size, n)
        
        for i in range(start_i, end_i):
            x_i = x[i]
            inner_sum = 0.0
            
            # Векторизованный внутренний цикл
            for j in range(n):
                w_ij = w[i, j]
                if w_ij != 0.0:
                    inner_sum = _fused_multiply_add(w_ij * x_i, (1.0 - x[j]), inner_sum)
            
            local_sum += inner_sum
        
        total += local_sum
    
    return total

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def _maxcut_sparse_csr_parallel(x: np.ndarray, csr_data: np.ndarray, csr_indices: np.ndarray, 
                                csr_indptr: np.ndarray) -> float64:
    """Ультра-быстрое вычисление для CSR-формата с параллелизацией."""
    n = len(x)
    total = 0.0
    
    for i in prange(n):
        x_i = x[i]
        row_sum = 0.0
        
        start_idx = csr_indptr[i]
        end_idx = csr_indptr[i + 1]
        
        # Векторизованная обработка строки
        for idx in range(start_idx, end_idx):
            j = csr_indices[idx]
            w_ij = csr_data[idx]
            row_sum = _fused_multiply_add(w_ij * x_i, (1.0 - x[j]), row_sum)
        
        total += row_sum
    
    return total

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def _convert_to_csr(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Быстрая конвертация в CSR формат."""
    n_rows, n_cols = w.shape
    
    # Подсчет ненулевых элементов в каждой строке
    row_counts = np.zeros(n_rows, dtype=int64)
    nnz = 0
    
    for i in range(n_rows):
        for j in range(n_cols):
            if w[i, j] != 0.0:
                row_counts[i] += 1
                nnz += 1
    
    # Создание CSR структуры
    data = np.zeros(nnz, dtype=float64)
    indices = np.zeros(nnz, dtype=int64)
    indptr = np.zeros(n_rows + 1, dtype=int64)
    
    # Заполнение indptr
    indptr[0] = 0
    for i in range(n_rows):
        indptr[i + 1] = indptr[i] + row_counts[i]
    
    # Заполнение data и indices
    current_pos = np.zeros(n_rows, dtype=int64)
    for i in range(n_rows):
        current_pos[i] = indptr[i]
    
    for i in range(n_rows):
        for j in range(n_cols):
            if w[i, j] != 0.0:
                pos = current_pos[i]
                data[pos] = w[i, j]
                indices[pos] = j
                current_pos[i] += 1
    
    return data, indices, indptr

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def _compute_advanced_sparsity_metrics(w: np.ndarray) -> Tuple[float64, float64, float64]:
    """Расширенные метрики разреженности для оптимального выбора алгоритма."""
    total_elements = w.size
    non_zero_count = 0
    max_row_density = 0.0
    variance_accumulator = 0.0
    
    n_rows, n_cols = w.shape
    
    for i in range(n_rows):
        row_nnz = 0
        for j in range(n_cols):
            if w[i, j] != 0.0:
                non_zero_count += 1
                row_nnz += 1
        
        row_density = row_nnz / n_cols
        if row_density > max_row_density:
            max_row_density = row_density
    
    sparsity_ratio = non_zero_count / total_elements
    
    # Вычисление дисперсии плотности строк
    mean_density = sparsity_ratio
    for i in range(n_rows):
        row_nnz = 0
        for j in range(n_cols):
            if w[i, j] != 0.0:
                row_nnz += 1
        row_density = row_nnz / n_cols
        variance_accumulator += (row_density - mean_density) ** 2
    
    density_variance = variance_accumulator / n_rows
    
    return sparsity_ratio, max_row_density, density_variance

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def _maxcut_symmetric_triangular_vectorized(x: np.ndarray, w: np.ndarray) -> float64:
    """Векторизованная оптимизация для симметричных матриц через верхний треугольник."""
    n = x.shape[0]
    total = 0.0
    
    for i in range(n):
        x_i = x[i]
        
        # Векторизованная обработка верхнего треугольника
        for j in range(i, n):
            w_ij = w[i, j]
            if w_ij != 0.0:
                contrib = _ultra_fast_contribution(x_i, x[j], w_ij)
                
                # Для несимметричных элементов удваиваем вклад
                if i != j:
                    contrib += _ultra_fast_contribution(x[j], x_i, w_ij)
                
                total += contrib
    
    return total

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def _is_symmetric_fast(w: np.ndarray, tolerance: float64 = 1e-12) -> boolean:
    """Быстрая проверка симметричности с ранним выходом."""
    n_rows, n_cols = w.shape
    if n_rows != n_cols:
        return False
    
    # Проверяем только верхний треугольник
    for i in range(n_rows):
        for j in range(i + 1, n_cols):
            if abs(w[i, j] - w[j, i]) > tolerance:
                return False
    
    return True

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def _optimal_tile_size(matrix_size: int64, cache_size: int64 = 32768) -> int64:
    """Вычисление оптимального размера тайла на основе размера кэша."""
    # Учитываем, что нужно поместить два вектора в кэш
    elements_per_tile = cache_size // (2 * 8)  # 8 байт на float64
    tile_size = int64(math.sqrt(elements_per_tile))
    
    # Ограничиваем размер тайла разумными пределами
    if tile_size < 16:
        tile_size = 16
    elif tile_size > 256:
        tile_size = 256
    
    return tile_size

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def _maxcut_batch_ultra_optimized(x_batch: np.ndarray, w: np.ndarray, 
                                  algorithm_type: int64) -> np.ndarray:
    """Ультра-оптимизированное батчевое вычисление с выбором алгоритма."""
    batch_size = x_batch.shape[0]
    results = np.zeros(batch_size, dtype=float64)
    
    # Предварительная подготовка для sparse матриц
    csr_data = np.empty(0, dtype=float64)
    csr_indices = np.empty(0, dtype=int64)
    csr_indptr = np.empty(0, dtype=int64)
    
    if algorithm_type == 1:  # Sparse CSR
        csr_data, csr_indices, csr_indptr = _convert_to_csr(w)
    
    tile_size = _optimal_tile_size(w.shape[0])
    
    for batch_idx in prange(batch_size):
        x = x_batch[batch_idx]
        
        if algorithm_type == 0:  # Dense tiled
            results[batch_idx] = _maxcut_parallel_tiled(x, w, tile_size)
        elif algorithm_type == 1:  # Sparse CSR
            results[batch_idx] = _maxcut_sparse_csr_parallel(x, csr_data, csr_indices, csr_indptr)
        else:  # Symmetric
            results[batch_idx] = _maxcut_symmetric_triangular_vectorized(x, w)
    
    return results

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def _maxcut_gradient_vectorized(x: np.ndarray, w: np.ndarray, epsilon: float64) -> np.ndarray:
    """Векторизованное вычисление градиента с параллелизацией."""
    n = x.shape[0]
    gradient = np.zeros(n, dtype=float64)
    
    # Базовое значение
    base_value = _maxcut_parallel_tiled(x, w, _optimal_tile_size(n))
    
    for i in prange(n):
        x_perturbed = x.copy()
        x_perturbed[i] = min(x_perturbed[i] + epsilon, 1.0)
        
        perturbed_value = _maxcut_parallel_tiled(x_perturbed, w, _optimal_tile_size(n))
        gradient[i] = (perturbed_value - base_value) / epsilon
    
    return gradient

def maxcut_obj(x: np.ndarray, w: np.ndarray) -> float:
    """Ультра-оптимизированное вычисление MaxCut с адаптивным выбором алгоритма.
    
    Применяет многоуровневую систему оптимизации:
    - Анализ продвинутых метрик разреженности и симметричности
    - Адаптивный выбор между CSR-sparse, tiled-dense и symmetric алгоритмами
    - Оптимизация размера тайлов под архитектуру процессора
    - Максимальное использование vectorization и FMA инструкций
    
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix returned by get_adjacency_matrix
    Returns:
        float: value of the cut.
    """
    # Валидация входных данных
    if x.size == 0 or w.size == 0:
        return 0.0
    
    if len(x) != w.shape[0] or w.shape[0] != w.shape[1]:
        raise ValueError("Несовместимые размеры x и w")
    
    # Подготовка данных для оптимальной работы с numba
    x_opt = np.ascontiguousarray(x, dtype=np.float64)
    w_opt = np.ascontiguousarray(w, dtype=np.float64)
    
    # Расширенный анализ характеристик матрицы
    sparsity_ratio, max_row_density, density_variance = _compute_advanced_sparsity_metrics(w_opt)
    matrix_size = w_opt.shape[0]
    
    # Продвинутая эвристика выбора алгоритма
    complexity_factor = matrix_size * matrix_size * sparsity_ratio
    
    # Выбор оптимального алгоритма на основе множественных критериев
    if (sparsity_ratio < SPARSE_THRESHOLD and 
        matrix_size > 200 and 
        density_variance < 0.1):
        # Высокоразреженная матрица с равномерным распределением
        csr_data, csr_indices, csr_indptr = _convert_to_csr(w_opt)
        return _maxcut_sparse_csr_parallel(x_opt, csr_data, csr_indices, csr_indptr)
    
    elif (_is_symmetric_fast(w_opt) and 
          matrix_size > SYMMETRIC_THRESHOLD and 
          sparsity_ratio > 0.3):
        # Симметричная матрица средней плотности
        return _maxcut_symmetric_triangular_vectorized(x_opt, w_opt)
    
    else:
        # Общий случай с тайловой оптимизацией
        optimal_tile = _optimal_tile_size(matrix_size)
        return _maxcut_parallel_tiled(x_opt, w_opt, optimal_tile)

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def maxcut_obj_batch(x_batch: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Ультра-оптимизированная батчевая версия maxcut_obj.
    
    Args:
        x_batch (numpy.ndarray): batch of binary strings, shape (batch_size, n_vertices)
        w (numpy.ndarray): adjacency matrix
    Returns:
        numpy.ndarray: array of cut values for each configuration
    """
    x_batch_opt = np.ascontiguousarray(x_batch, dtype=np.float64)
    w_opt = np.ascontiguousarray(w, dtype=np.float64)
    
    # Анализ для выбора алгоритма
    sparsity_ratio, _, density_variance = _compute_advanced_sparsity_metrics(w_opt)
    matrix_size = w_opt.shape[0]
    
    # Определение типа алгоритма
    if sparsity_ratio < SPARSE_THRESHOLD and matrix_size > 200:
        algorithm_type = int64(1)  # Sparse CSR
    elif _is_symmetric_fast(w_opt) and matrix_size > SYMMETRIC_THRESHOLD:
        algorithm_type = int64(2)  # Symmetric
    else:
        algorithm_type = int64(0)  # Dense tiled
    
    return _maxcut_batch_ultra_optimized(x_batch_opt, w_opt, algorithm_type)

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def maxcut_obj_gradient_approx(x: np.ndarray, w: np.ndarray, epsilon: float64 = 1e-8) -> np.ndarray:
    """Ультра-быстрое приближенное вычисление градиента MaxCut.
    
    Args:
        x (numpy.ndarray): current solution
        w (numpy.ndarray): adjacency matrix  
        epsilon (float): finite difference step
    Returns:
        numpy.ndarray: approximate gradient
    """
    x_opt = np.ascontiguousarray(x, dtype=np.float64)
    w_opt = np.ascontiguousarray(w, dtype=np.float64)
    
    return _maxcut_gradient_vectorized(x_opt, w_opt, epsilon)

# Дополнительные функции для специальных случаев

@jit(nopython=True, cache=True, fastmath=True, parallel=True, boundscheck=False)
def maxcut_obj_binary_optimized(x_binary: np.ndarray, w: np.ndarray) -> float64:
    """Специализированная версия для строго бинарных векторов (0/1).
    
    Использует битовые операции для максимального ускорения.
    """
    n = x_binary.shape[0]
    total = 0.0
    
    for i in prange(n):
        if x_binary[i] == 1:
            local_sum = 0.0
            for j in range(n):
                if x_binary[j] == 0 and w[i, j] != 0.0:
                    local_sum += w[i, j]
            total += local_sum
    
    return total

@jit(nopython=True, cache=True, fastmath=True, boundscheck=False)
def maxcut_obj_incremental_update(x: np.ndarray, w: np.ndarray, 
                                  current_value: float64, flip_index: int64) -> float64:
    """Инкрементальное обновление значения MaxCut при изменении одного бита.
    
    Для алгоритмов локального поиска - O(n) вместо O(n²).
    """
    n = x.shape[0]
    delta = 0.0
    
    old_value = x[flip_index]
    new_value = 1.0 - old_value
    
    for j in range(n):
        if j != flip_index and w[flip_index, j] != 0.0:
            # Изменение вклада при переключении бита
            old_contrib = _ultra_fast_contribution(old_value, x[j], w[flip_index, j])
            new_contrib = _ultra_fast_contribution(new_value, x[j], w[flip_index, j])
            delta += new_contrib - old_contrib
            
            # Симметричный вклад
            old_contrib_sym = _ultra_fast_contribution(x[j], old_value, w[j, flip_index])
            new_contrib_sym = _ultra_fast_contribution(x[j], new_value, w[j, flip_index])
            delta += new_contrib_sym - old_contrib_sym
    
    return current_value + delta