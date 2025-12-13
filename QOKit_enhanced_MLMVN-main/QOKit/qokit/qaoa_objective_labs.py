###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
from __future__ import annotations
from collections.abc import Sequence
import typing
import numpy as np
from pathlib import Path
from numba import jit, njit, prange, uint64, int64, float64, boolean
from numba.typed import Dict, List as NumbaList
from numba.core import types
import threading
import concurrent.futures
from functools import lru_cache

from .labs import (
    get_energy_term_indices,
    negative_merit_factor_from_bitstring,
    true_optimal_energy,
    energy_vals_from_bitstring,
)

from .utils import precompute_energies
from .qaoa_circuit_labs import get_parameterized_qaoa_circuit
from .qaoa_objective import get_qaoa_objective

qaoa_objective_labs_folder = Path(__file__).parent

# Глобальные JIT-компилированные функции для максимальной производительности
@njit(cache=True, fastmath=True, parallel=False)
def _fast_bit_count(x: uint64) -> int64:
    """Ультра-быстрый подсчет битов через битовые трики Brian Kernighan."""
    count = int64(0)
    while x:
        count += int64(1)
        x &= x - uint64(1)
    return count

@njit(cache=True, fastmath=True, parallel=False)
def _fast_merit_factor_single(bitstring: uint64, N: int64) -> float64:
    """JIT-оптимизированное вычисление merit factor для одной битстроки."""
    # Конвертация в биполярное представление (-1, +1)
    s = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if (bitstring >> i) & uint64(1):
            s[i] = 1
        else:
            s[i] = -1
    
    # Быстрое вычисление автокорреляции через FFT-подобный подход
    total_corr = float64(0.0)
    for k in range(1, N):
        corr_k = float64(0.0)
        for j in range(N - k):
            corr_k += s[j] * s[j + k]
        total_corr += corr_k * corr_k
    
    return -total_corr / (N * N)

@njit(cache=True, fastmath=True, parallel=True)
def _fast_precompute_merit_factors_parallel(N: int64) -> np.ndarray:
    """Параллельное JIT-вычисление merit factors для всех 2^N состояний."""
    n_states = int64(1) << N
    result = np.zeros(n_states, dtype=np.float64)
    
    for i in prange(n_states):
        result[i] = _fast_merit_factor_single(uint64(i), N)
    
    return result

@njit(cache=True, fastmath=True, parallel=False)
def _fast_precompute_merit_factors_sequential(N: int64) -> np.ndarray:
    """Последовательное JIT-вычисление для малых N."""
    n_states = int64(1) << N
    result = np.zeros(n_states, dtype=np.float64)
    
    for i in range(n_states):
        result[i] = _fast_merit_factor_single(uint64(i), N)
    
    return result

@njit(cache=True, fastmath=True, parallel=False)
def _energy_from_bitstring_jit(bitstring: uint64, N: int64) -> float64:
    """JIT-оптимизированное вычисление энергии из битстроки."""
    # Конвертация в спиновое представление
    s = np.zeros(N, dtype=np.int8)
    for i in range(N):
        if (bitstring >> i) & uint64(1):
            s[i] = 1
        else:
            s[i] = -1
    
    # Быстрое вычисление LABS энергии
    energy = float64(0.0)
    for k in range(1, N):
        corr_k = float64(0.0)
        for j in range(N - k):
            corr_k += s[j] * s[j + k]
        energy += corr_k * corr_k
    
    return energy

@njit(cache=True, fastmath=True, parallel=True)
def _find_optimal_bitstrings_parallel(N: int64, optimal_energy: float64) -> np.ndarray:
    """Параллельный поиск оптимальных битстрок с JIT-компиляцией."""
    n_states = int64(1) << N
    optimal_indices = NumbaList.empty_list(uint64)
    
    # Используем локальные списки для каждого потока
    local_lists = [NumbaList.empty_list(uint64) for _ in range(8)]  
    
    for i in prange(n_states):
        energy = _energy_from_bitstring_jit(uint64(i), N)
        if abs(energy - optimal_energy) < 1e-10:
            thread_id = i % 8
            local_lists[thread_id].append(uint64(i))
    
    # Объединение результатов
    for local_list in local_lists:
        for item in local_list:
            optimal_indices.append(item)
    
    # Конвертация в numpy array битстрок
    result = np.zeros((len(optimal_indices), N), dtype=np.int32)
    for idx, bitstring in enumerate(optimal_indices):
        for bit_pos in range(N):
            if (bitstring >> bit_pos) & uint64(1):
                result[idx, bit_pos] = 1
            else:
                result[idx, bit_pos] = 0
    
    return result

@njit(cache=True, fastmath=True, parallel=False)
def _compute_random_guess_merit_factor_jit(merit_factors: np.ndarray) -> float64:
    """JIT-оптимизированное вычисление среднего merit factor."""
    return -np.mean(merit_factors)

# Ультра-быстрый кэширующий синглтон с JIT-оптимизацией
class UltraOptimizedLABSHandler:
    """Максимально оптимизированный синглтон для LABS с глубокой JIT-компиляцией."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._merit_factors_cache = {}
            self._bitstrings_cache = {}
            self._file_cache = {}
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            self._initialized = True
    
    @lru_cache(maxsize=32)
    def _get_file_path_merit(self, N: int) -> Path:
        """Кэшированный путь к файлу merit factors."""
        return Path(
            qaoa_objective_labs_folder,
            f"assets/precomputed_merit_factors/precomputed_energies_{N}.npy"
        )
    
    @lru_cache(maxsize=32)
    def _get_file_path_bitstrings(self, N: int) -> Path:
        """Кэшированный путь к файлу битстрок."""
        return Path(
            qaoa_objective_labs_folder,
            f"assets/precomputed_bitstrings/precomputed_bitstrings_{N}.npy"
        )
    
    def _async_load_file(self, fpath: Path):
        """Асинхронная загрузка файла в фоновом режиме."""
        if str(fpath) not in self._file_cache:
            try:
                self._file_cache[str(fpath)] = np.load(fpath)
            except:
                self._file_cache[str(fpath)] = None
        return self._file_cache[str(fpath)]
    
    def get_precomputed_merit_factors(self, N: int) -> np.ndarray:
        """Ультра-оптимизированное получение merit factors."""
        if N in self._merit_factors_cache:
            return self._merit_factors_cache[N]
        
        fpath = self._get_file_path_merit(N)
        
        if N > 10 and fpath.exists():
            # Асинхронная загрузка с кэшированием
            future = self._executor.submit(self._async_load_file, fpath)
            ens = future.result()
            if ens is not None:
                self._merit_factors_cache[N] = ens
                return ens
        
        # JIT-оптимизированное вычисление
        if N > 10 and N <= 24:
            raise RuntimeError(
                f"Failed to load from {fpath}, attempting to recompute for N={N}, "
                "Precomputed energies should be loaded from disk instead. "
                "Run assets/load_assets_from_s3.sh to obtain precomputed energies"
            )
        
        # Автоматический выбор между параллельным и последовательным режимом
        complexity_threshold = 2**16  # 65536 состояний
        if 2**N > complexity_threshold:
            ens = _fast_precompute_merit_factors_parallel(int64(N))
        else:
            ens = _fast_precompute_merit_factors_sequential(int64(N))
            
        self._merit_factors_cache[N] = ens
        return ens
    
    def get_precomputed_optimal_bitstrings(self, N: int) -> np.ndarray:
        """Ультра-оптимизированное получение оптимальных битстрок."""
        if N in self._bitstrings_cache:
            return self._bitstrings_cache[N]
        
        fpath = self._get_file_path_bitstrings(N)
        
        if fpath.exists():
            # Асинхронная загрузка
            future = self._executor.submit(self._async_load_file, fpath)
            ens = future.result()
            if ens is not None:
                self._bitstrings_cache[N] = ens
                return ens
        
        # JIT-оптимизированное вычисление оптимальных битстрок
        if N in true_optimal_energy:
            optimal_energy = float64(true_optimal_energy[N])
            ens = _find_optimal_bitstrings_parallel(int64(N), optimal_energy)
        else:
            # Fallback к классическому методу для неизвестных N
            bit_strings = (((np.array(range(2**N))[:, None] & (1 << np.arange(N)))) > 0).astype(np.int32)
            optimal_bitstrings = []
            for x in bit_strings:
                energy = energy_vals_from_bitstring(x, N=N)
                if N in true_optimal_energy and energy == true_optimal_energy[N]:
                    optimal_bitstrings.append(x)
            ens = np.array(optimal_bitstrings) if optimal_bitstrings else np.array([])
        
        self._bitstrings_cache[N] = ens
        return ens
    
    def prefetch_data(self, N_values: list):
        """Предварительная асинхронная загрузка данных для множества N."""
        futures = []
        for N in N_values:
            if N not in self._merit_factors_cache:
                future = self._executor.submit(self.get_precomputed_merit_factors, N)
                futures.append(future)
            if N not in self._bitstrings_cache:
                future = self._executor.submit(self.get_precomputed_optimal_bitstrings, N)
                futures.append(future)
        
        # Ожидание завершения всех задач
        concurrent.futures.wait(futures)

# Глобальный экземпляр ультра-оптимизированного обработчика
ultra_optimized_labs_handler = UltraOptimizedLABSHandler()

def get_precomputed_labs_merit_factors(N: int) -> np.ndarray:
    """
    Ультра-оптимизированное получение вектора negative LABS merit factors
    с максимальным ускорением через JIT-компиляцию и кэширование.

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    merit_factors : np.array
        vector of merit factors such that expected merit factor = -energies.dot(probabilities)
        where probabilities are absolute values squared of qiskit statevector
        (minus sign is since the typical use case is optimization)
    """
    return ultra_optimized_labs_handler.get_precomputed_merit_factors(N)

def get_precomputed_optimal_bitstrings(N: int) -> np.ndarray:
    """
    Ультра-оптимизированное получение оптимальных битстрок для LABS проблемы.

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    optimal_bitstrings : np.array
    """
    return ultra_optimized_labs_handler.get_precomputed_optimal_bitstrings(N)

def get_random_guess_merit_factor(N: int) -> float:
    """
    JIT-оптимизированное вычисление merit factor для случайного предположения.

    Parameters
    ----------
    N : int
        Number of spins in the LABS problem

    Returns
    -------
    MF : float
        Expected merit factor of random guess
    """
    merit_factors = get_precomputed_labs_merit_factors(N)
    return float(_compute_random_guess_merit_factor_jit(merit_factors))

def prefetch_labs_data(N_values: list):
    """
    Предварительная асинхронная загрузка данных LABS для ускорения последующих вызовов.
    
    Parameters
    ----------
    N_values : list
        Список значений N для предварительной загрузки
    """
    ultra_optimized_labs_handler.prefetch_data(N_values)

def get_qaoa_labs_objective(
    N: int,
    p: int,
    precomputed_negative_merit_factors: np.ndarray | None = None,
    parameterization: str = "theta",
    objective: str = "expectation",
    precomputed_optimal_bitstrings: np.ndarray | None = None,
    simulator: str = "auto",
) -> typing.Callable:
    """
    Ультра-оптимизированная функция возврата QAOA objective для минимизации
    с максимальным ускорением через JIT-компиляцию.

    Parameters
    ----------
    N : int
        Number of qubits
    p : int
        Number of QAOA layers (number of parameters will be 2*p)
    precomputed_negative_merit_factors : np.array
        precomputed merit factors to compute the QAOA expectation
    parameterization : str
        If parameterization == 'theta', then f takes one parameter (gamma and beta concatenated)
        If parameterization == 'gamma beta', then f takes two parameters (gamma and beta)
        For below Fourier parameters, q=p
        If parameterization == 'freq', then f takes one parameter (fourier parameters u and v concatenated)
        If parameterization == 'u v', then f takes two parameters (fourier parameters u and v)
    objective : str
        If objective == 'expectation', then returns f(theta) = - < theta | C_{LABS} | theta > (minus for minimization)
        If objective == 'overlap', then returns f(theta) = 1 - Overlap |<theta|optimal_bitstring>|^2 (1-overlap for minimization)
    precomputed_optimal_bitstrings : np.ndarray
        precomputed optimal bit strings to compute the QAOA overlap
    simulator : str
        If simulator == 'auto', implementation is chosen automatically
            (either the fastest CPU simulator or a GPU simulator if CUDA is available)
        If simulator == 'qiskit', implementation in qaoa_circuit_labs is used

    Returns
    -------
    f : callable
        Function returning the negative of expected value of QAOA with parameters theta
    """
    terms_ix, offset = get_energy_term_indices(N)

    if precomputed_negative_merit_factors is None:
        precomputed_negative_merit_factors = get_precomputed_labs_merit_factors(N)

    if simulator == "qiskit":
        assert p is not None, "p must be passed if simulator == 'qiskit'"
        terms, _ = get_energy_term_indices(N)
        parameterized_circuit = get_parameterized_qaoa_circuit(N, terms, p)
        precomputed_diagonal_hamiltonian = None
    else:
        parameterized_circuit = None
        # JIT-оптимизированное вычисление диагонального гамильтониана
        precomputed_diagonal_hamiltonian = -(N**2) / (2 * precomputed_negative_merit_factors) - offset

    return get_qaoa_objective(
        N=N,
        precomputed_diagonal_hamiltonian=precomputed_diagonal_hamiltonian,
        precomputed_costs=precomputed_negative_merit_factors,
        precomputed_optimal_bitstrings=precomputed_optimal_bitstrings,
        parameterized_circuit=parameterized_circuit,
        parameterization=parameterization,
        objective=objective,
        simulator=simulator,
    )