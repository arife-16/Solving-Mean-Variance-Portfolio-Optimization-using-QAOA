from __future__ import annotations
import sys
import numpy as np
from itertools import combinations
from numba import njit, jit, prange, types, typed, uint64, int64, float64
from numba.core import config
from numba.extending import overload_method, intrinsic
from numba.core.types import Array, DictType, float64 as nb_float64, int64 as nb_int64
from numba import types as nb_types
from numba.core import cgutils
from numba.core.imputils import impl_ret_untracked
from numba.core.types import Tuple, UniTuple
from numba.core.errors import NumbaTypeSafetyWarning
from typing import Optional, Tuple, List, Sequence, Iterable
import math
import warnings

# Подавление всех предупреждений
warnings.filterwarnings("ignore")

# Максимальная оптимизация JIT
config.DISABLE_JIT = False
config.NUMBA_BOUNDSCHECK = False
config.NUMBA_DEBUG_FRONTEND = False
config.NUMBA_DISABLE_ERROR_MESSAGE_HIGHLIGHTING = True
config.NUMBA_CAPTURED_ERRORS = 'old_style'
config.NUMBA_FULL_TRACEBACKS = False
config.NUMBA_ENABLE_AVX = True
config.NUMBA_CPU_NAME = 'native'
config.NUMBA_CPU_FEATURES = 'native'
config.NUMBA_DISABLE_INTEL_SVML = False

# Оптимизированные константы
SIMD_WIDTH = 8
AVX_WIDTH = 16
PARALLEL_THRESHOLD = 64
VECTORIZATION_THRESHOLD = 32
CACHE_TILE_SIZE = 128
NUMA_CHUNK_SIZE = 512
LARGE_NUMBER = 1e15
UNROLL_FACTOR = 16

@intrinsic
def fast_fma_intrinsic(typingctx, a_type, b_type, c_type):
    """Прямое использование FMA инструкций процессора для максимальной скорости."""
    sig = nb_float64(nb_float64, nb_float64, nb_float64)
    
    def fast_fma_impl(context, builder, sig, args):
        [a, b, c] = args
        fma_intr = cgutils.get_or_insert_function(
            builder.module, 
            types.Function(types.DoubleType(), [types.DoubleType(), types.DoubleType(), types.DoubleType()]),
            "llvm.fma.f64"
        )
        result = builder.call(fma_intr, [a, b, c])
        return result
    
    return sig, fast_fma_impl

@njit(cache=True, fastmath=True, inline='always', boundscheck=False, 
      error_model='numpy', forceinline=True)
def _fma(a: float64, b: float64, c: float64) -> float64:
    """Оптимизированное FMA."""
    return fast_fma_intrinsic(a, b, c)

@njit(cache=True, fastmath=True, inline='always', boundscheck=False,
      error_model='numpy', forceinline=True)
def _vectorized_autocorr_core(s: np.ndarray, k: int64, start: int64, end: int64) -> float64:
    """Максимально оптимизированная автокорреляция с векторизацией."""
    accumulator = 0.0
    i = start
    
    # AVX векторизация - обработка по 16 элементов
    while i + UNROLL_FACTOR <= end and i + k + UNROLL_FACTOR <= len(s):
        local_sum = 0.0
        for unroll_idx in range(UNROLL_FACTOR):
            idx = i + unroll_idx
            local_sum = _fma(s[idx], s[idx + k], local_sum)
        accumulator += local_sum
        i += UNROLL_FACTOR
    
    # SIMD векторизация - обработка по 8 элементов
    while i + SIMD_WIDTH <= end and i + k + SIMD_WIDTH <= len(s):
        local_sum = 0.0
        for unroll_idx in range(SIMD_WIDTH):
            idx = i + unroll_idx
            local_sum = _fma(s[idx], s[idx + k], local_sum)
        accumulator += local_sum
        i += SIMD_WIDTH
    
    # Обработка по 4 элемента
    while i + 4 <= end and i + k + 4 <= len(s):
        accumulator = _fma(s[i], s[i + k], accumulator)
        accumulator = _fma(s[i + 1], s[i + 1 + k], accumulator)
        accumulator = _fma(s[i + 2], s[i + 2 + k], accumulator)
        accumulator = _fma(s[i + 3], s[i + 3 + k], accumulator)
        i += 4
    
    # Оставшиеся элементы
    while i < end and i + k < len(s):
        accumulator = _fma(s[i], s[i + k], accumulator)
        i += 1
    
    return accumulator

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy', nogil=True)
def _algorithm_selector(N: int64) -> int64:
    """Выбор оптимального алгоритма на основе размера."""
    if N <= VECTORIZATION_THRESHOLD:
        return int64(0)  # Базовый алгоритм
    elif N <= 200:
        return int64(1)  # Параллельный алгоритм
    else:
        return int64(1)  # Параллельный для больших N

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy', nogil=True)
def _compute_energy_basic(s: np.ndarray, N: int64) -> float64:
    """Базовый алгоритм с максимальной векторизацией."""
    E_s = 0.0
    
    for k in range(1, N):
        C_k = 0.0
        max_i = N - k
        i = 0
        
        # Максимальная векторизация с развертыванием
        while i + UNROLL_FACTOR <= max_i:
            local_sum = 0.0
            for unroll_offset in range(UNROLL_FACTOR):
                idx = i + unroll_offset
                local_sum = _fma(s[idx], s[idx + k], local_sum)
            C_k += local_sum
            i += UNROLL_FACTOR
        
        while i + SIMD_WIDTH <= max_i:
            local_sum = 0.0
            for unroll_offset in range(SIMD_WIDTH):
                idx = i + unroll_offset
                local_sum = _fma(s[idx], s[idx + k], local_sum)
            C_k += local_sum
            i += SIMD_WIDTH
        
        while i + 4 <= max_i:
            C_k = _fma(s[i], s[i + k], C_k)
            C_k = _fma(s[i + 1], s[i + 1 + k], C_k)
            C_k = _fma(s[i + 2], s[i + 2 + k], C_k)
            C_k = _fma(s[i + 3], s[i + 3 + k], C_k)
            i += 4
        
        while i < max_i:
            C_k = _fma(s[i], s[i + k], C_k)
            i += 1
        
        E_s = _fma(C_k, C_k, E_s)
    
    return E_s

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy', 
      nogil=True, parallel=True)
def _compute_energy_parallel(s: np.ndarray, N: int64) -> float64:
    """Параллельный алгоритм с NUMA-оптимизацией."""
    total_energy = 0.0
    
    for k in prange(1, N):
        max_i = N - k
        tile_size = min(CACHE_TILE_SIZE, max_i)
        n_tiles = (max_i + tile_size - 1) // tile_size
        k_energy = 0.0
        
        for tile_idx in range(n_tiles):
            start_idx = tile_idx * tile_size
            end_idx = min(start_idx + tile_size, max_i)
            tile_corr = _vectorized_autocorr_core(s, k, start_idx, end_idx)
            k_energy += tile_corr
        
        total_energy += k_energy * k_energy
    
    return total_energy

@njit(cache=True, fastmath=True, boundscheck=False, error_model='numpy', nogil=True)
def _compute_energy_core(s: np.ndarray, N: int64) -> float64:
    """Ядро вычисления энергии с автоматическим выбором алгоритма."""
    algorithm_type = _algorithm_selector(N)
    
    if algorithm_type == 0:
        return _compute_energy_basic(s, N)
    else:
        return _compute_energy_parallel(s, N)

@njit(cache=True, fastmath=True, parallel=True, boundscheck=False,
      error_model='numpy', nogil=True)
def _batch_energy_computation(s_batch: np.ndarray, N: int64) -> np.ndarray:
    """Батчевое вычисление энергий."""
    batch_size = s_batch.shape[0]
    results = np.zeros(batch_size, dtype=float64)
    
    for batch_idx in prange(batch_size):
        s = s_batch[batch_idx]
        results[batch_idx] = _compute_energy_core(s, N)
    
    return results

@njit(cache=True, fastmath=True, parallel=True, boundscheck=False,
      error_model='numpy', nogil=True)
def _compute_autocorr_spectrum(s: np.ndarray, N: int64) -> np.ndarray:
    """Вычисление спектра автокорреляций."""
    autocorrs = np.zeros(N - 1, dtype=float64)
    
    for k in prange(1, N):
        C_k = 0.0
        max_i = N - k
        i = 0
        
        while i + SIMD_WIDTH <= max_i:
            local_sum = 0.0
            for unroll_idx in range(SIMD_WIDTH):
                local_sum = _fma(s[i + unroll_idx], s[i + unroll_idx + k], local_sum)
            C_k += local_sum
            i += SIMD_WIDTH
        
        while i < max_i:
            C_k = _fma(s[i], s[i + k], C_k)
            i += 1
        
        autocorrs[k - 1] = C_k
    
    return autocorrs

@njit(cache=True, fastmath=True, parallel=True, boundscheck=False,
      error_model='numpy', nogil=True)
def _batch_autocorr_spectra(s_batch: np.ndarray, N: int64) -> np.ndarray:
    """Батчевое вычисление спектров автокорреляций."""
    batch_size = s_batch.shape[0]
    autocorrs_batch = np.zeros((batch_size, N - 1), dtype=float64)
    
    for batch_idx in prange(batch_size):
        s = s_batch[batch_idx]
        autocorrs_batch[batch_idx] = _compute_autocorr_spectrum(s, N)
    
    return autocorrs_batch

# Основные функции API

def energy_vals(s: Sequence, N: int | None = None) -> float:
    """Оптимизированное вычисление энергии LABS."""
    if N is None:
        N = len(s)
    
    if N <= 0:
        return 0.0
    
    s_opt = np.ascontiguousarray(s, dtype=np.float64)
    return _compute_energy_core(s_opt, int64(N))

def energy_vals_from_bitstring(x, N: int | None = None) -> float:
    """Вычисление энергии из битовой строки."""
    s_transformed = 1 - 2 * np.asarray(x, dtype=np.float64)
    return energy_vals(s_transformed, N=N)

def merit_factor(s: Sequence, N: int | None = None) -> float:
    """Вычисление merit factor."""
    if N is None:
        N = len(s)
    
    if N <= 0:
        return 0.0
    
    E_s = energy_vals(s, N=N)
    return N**2 / (2 * E_s) if E_s > 0 else float(LARGE_NUMBER)

def negative_merit_factor_from_bitstring(x, N: int | None = None) -> float:
    """Негативная merit factor для оптимизации."""
    mf = merit_factor(1 - 2 * np.asarray(x, dtype=np.float64), N=N)
    return -mf

def energy_vals_batch(s_batch: np.ndarray, N: int | None = None) -> np.ndarray:
    """Батчевое вычисление энергий."""
    if N is None:
        N = s_batch.shape[1]
    
    return _batch_energy_computation(s_batch, int64(N))

def merit_factor_batch(s_batch: np.ndarray, N: int | None = None) -> np.ndarray:
    """Батчевое вычисление merit factors."""
    energies = energy_vals_batch(s_batch, N)
    if N is None:
        N = s_batch.shape[1]
    
    N_squared_half = (N * N) / 2.0
    merit_factors = np.zeros_like(energies)
    
    for i in range(len(energies)):
        energy_val = energies[i]
        if energy_val > 0:
            merit_factors[i] = N_squared_half / energy_val
        else:
            merit_factors[i] = float(LARGE_NUMBER)
    
    return merit_factors

def get_autocorrelation_spectrum(s: Sequence, N: int | None = None) -> np.ndarray:
    """Получение спектра автокорреляций."""
    if N is None:
        N = len(s)
    
    if N <= 1:
        return np.array([])
    
    s_opt = np.ascontiguousarray(s, dtype=np.float64)
    return _compute_autocorr_spectrum(s_opt, int64(N))

def get_batch_autocorrelation_spectra(s_batch: np.ndarray, N: int | None = None) -> np.ndarray:
    """Батчевое получение спектров автокорреляций."""
    if N is None:
        N = s_batch.shape[1]
    
    if N <= 1:
        return np.zeros((s_batch.shape[0], 0))
    
    return _batch_autocorr_spectra(s_batch, int64(N))

def energy_vals_general(s: Sequence, terms: Iterable | None = None, 
                       offset: float | None = None, check_parameters: bool = True) -> float:
    """Общая версия с термами (для совместимости)."""
    if terms is None or offset is None:
        return energy_vals(s)
    
    if check_parameters:
        assert set(s).issubset(set([-1, 1]))
    
    s_opt = np.ascontiguousarray(s, dtype=np.float64)
    E_s = float(offset)
    terms_list = list(terms) if not isinstance(terms, list) else terms
    
    for term in terms_list:
        term_product = 1.0
        for idx in term:
            term_product *= s_opt[idx]
        coefficient = 4.0 if len(term) == 4 else 2.0
        E_s += coefficient * term_product
    
    return E_s

def energy_vals_from_bitstring_general(x, terms: Sequence | None = None, 
                                     offset: float | None = None, 
                                     check_parameters: bool = False) -> float:
    """Общая версия для битовых строк."""
    s_transformed = 1 - 2 * np.asarray(x, dtype=np.float64)
    return energy_vals_general(s_transformed, terms=terms, offset=offset, 
                              check_parameters=check_parameters)

def get_energy_term_indices(N: int) -> Tuple[set, int]:
    """Получение индексов термов энергии."""
    if N <= 0:
        return set(), 0
    
    all_terms = set()
    offset = 0
    
    for k in range(1, N):
        offset += N - k
        for i, j in combinations(range(1, N - k + 1), 2):
            if i + k == j:
                all_terms.add(tuple(sorted((i - 1, j + k - 1))))
            else:
                all_terms.add(tuple(sorted((i - 1, i + k - 1, j - 1, j + k - 1))))
    
    return all_terms, offset

def slow_merit_factor(s: Sequence, terms: Iterable | None = None, 
                     offset: float | None = None, check_parameters: bool = True) -> float:
    """Merit factor с термами (для совместимости)."""
    E_s = energy_vals_general(s, terms=terms, offset=offset, 
                             check_parameters=check_parameters)
    N = len(s)
    return N**2 / (2 * E_s) if E_s > 0 else float(LARGE_NUMBER)

# Экспорт основных функций
__all__ = [
    'energy_vals', 'energy_vals_from_bitstring', 'merit_factor',
    'negative_merit_factor_from_bitstring', 'energy_vals_batch', 'merit_factor_batch',
    'get_autocorrelation_spectrum', 'get_batch_autocorrelation_spectra',
    'energy_vals_general', 'energy_vals_from_bitstring_general',
    'get_energy_term_indices', 'slow_merit_factor'
]