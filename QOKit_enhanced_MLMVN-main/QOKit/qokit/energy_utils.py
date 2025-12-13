import numpy as np
from itertools import product
from multiprocessing import Pool
from numba import jit, prange, uint64, int64, float64, boolean
from numba.typed import List as NumbaList
import numba


@jit(nopython=True, cache=True, fastmath=True)
def _generate_bit_strings_jit(num_variables: int):
    """JIT-оптимизированная генерация битовых строк"""
    total_combinations = 2**num_variables
    result = np.zeros((total_combinations, num_variables), dtype=np.int32)
    
    for i in range(total_combinations):
        for j in range(num_variables):
            result[i, j] = (i >> j) & 1
    
    return result


@jit(nopython=True, cache=True, fastmath=True)  
def _precompute_energies_jit_core(bit_strings, obj_func_vectorized):
    """Ядро для JIT-оптимизированного предвычисления энергий"""
    n_strings = bit_strings.shape[0]
    energies = np.zeros(n_strings, dtype=np.float64)
    
    for i in range(n_strings):
        energies[i] = obj_func_vectorized(bit_strings[i])
    
    return energies


@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def _precompute_energies_parallel_jit(bit_strings, obj_func_vectorized):
    """Параллельное JIT-оптимизированное предвычисление энергий"""
    n_strings = bit_strings.shape[0]
    energies = np.zeros(n_strings, dtype=np.float64)
    
    for i in prange(n_strings):
        energies[i] = obj_func_vectorized(bit_strings[i])
    
    return energies


@jit(nopython=True, cache=True, fastmath=True)
def _obj_from_statevector_jit(sv_abs_squared, precomputed_energies):
    """JIT-оптимизированное вычисление объективной функции из вектора состояний"""
    return np.dot(precomputed_energies, sv_abs_squared)


@jit(nopython=True, cache=True, fastmath=True)
def _brute_force_optimization_jit(bit_strings, obj_func_vectorized, minimize_flag, function_takes_spins):
    """JIT-оптимизированная брутфорс оптимизация"""
    n_strings = bit_strings.shape[0]
    
    if minimize_flag:
        best_cost = np.inf
    else:
        best_cost = -np.inf
    
    best_idx = 0
    
    for i in range(n_strings):
        x = bit_strings[i]
        
        if function_takes_spins:
            spins = 1 - 2 * x
            cost = obj_func_vectorized(spins)
        else:
            cost = obj_func_vectorized(x)
        
        if minimize_flag:
            if cost < best_cost:
                best_cost = cost
                best_idx = i
        else:
            if cost > best_cost:
                best_cost = cost
                best_idx = i
    
    return best_cost, best_idx


@jit(nopython=True, cache=True, fastmath=True)
def _objective_from_counts_jit(counts_keys, counts_values, obj_func_jit):
    """JIT-оптимизированная версия для вычисления среднего значения из счетчиков"""
    total_mean = 0.0
    total_counts = 0
    
    for i in range(len(counts_keys)):
        key = counts_keys[i]
        count = counts_values[i]
        
        obj_value = obj_func_jit(key)
        total_mean += obj_value * count
        total_counts += count
    
    return total_mean / total_counts


@jit(nopython=True, cache=True, fastmath=True)
def _yield_bitstring_jit(i, nbits):
    """JIT-оптимизированная генерация одной битовой строки"""
    result = np.zeros(nbits, dtype=np.int32)
    for j in range(nbits):
        result[j] = (i >> j) & 1
    return result


class JITObjectiveWrapper:
    """Обертка для создания JIT-совместимых объективных функций"""
    def __init__(self, obj_f, args, kwargs):
        self.obj_f = obj_f
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, x):
        return self.obj_f(x, *self.args, **self.kwargs)


def generate_bit_strings(num_variables: int):
    """
    Generate all possible bit strings for given number of variables
    Returns bit strings as numpy array where each row is a bit string
    """
    if num_variables <= 20:  # JIT эффективен для небольших размеров
        return _generate_bit_strings_jit(num_variables)
    else:
        # Векторизованная версия для больших размеров
        return (((np.array(range(2**num_variables), dtype=np.int64)[:, None] & (1 << np.arange(num_variables)))) > 0).astype(np.int32)


def yield_all_bitstrings(nbits: int):
    """
    Helper function to avoid having to store all bitstrings in memory
    nbits : int
        Number of parameters obj_f takes
    """
    for i in range(2**nbits):
        yield _yield_bitstring_jit(i, nbits)


def precompute_energies(obj_f, nbits: int, *args: object, **kwargs: object):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector

    For LABS-specific, accelerated version see get_precomputed_labs_merit_factors in qaoa_objective_labs.py

    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    *args, **kwargs : Object
        Parameters to be passed directly to obj_f

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector
    """
    bit_strings = generate_bit_strings(nbits)
    
    # Автоматическое решение о параллелизации
    complexity = 2**nbits
    use_parallel = complexity > 1000
    
    # Попытка JIT-оптимизации
    if not args and not kwargs:
        try:
            jit_obj_f = jit(nopython=True, cache=True, fastmath=True)(obj_f)
            
            if use_parallel:
                return _precompute_energies_parallel_jit(bit_strings, jit_obj_f)
            else:
                return _precompute_energies_jit_core(bit_strings, jit_obj_f)
                
        except:
            pass
    
    if args or kwargs:
        try:
            wrapper = JITObjectiveWrapper(obj_f, args, kwargs)
            jit_wrapper = jit(nopython=True, cache=True, fastmath=True)(wrapper)
            
            if use_parallel:
                return _precompute_energies_parallel_jit(bit_strings, jit_wrapper)
            else:
                return _precompute_energies_jit_core(bit_strings, jit_wrapper)
        except:
            pass
    
    # Fallback к стандартной реализации
    return np.array([obj_f(x, *args, **kwargs) for x in bit_strings], dtype=np.float64)


def precompute_energies_parallel(obj_f, nbits: int, num_processes: int, postfix: list = []):
    """
    Precomputed a vector of objective function values
    that accelerates the energy computation in obj_from_statevector
    Uses multiprocessing.Pool

    For LABS-specific, accelerated version see get_precomputed_labs_merit_factors in qaoa_objective_labs.py

    Parameters
    ----------
    obj_f : callable
        Objective function to precompute
    nbits : int
        Number of parameters obj_f takes
    num_processes : int
        Number of processes to use in multiprocessing.Pool
    postfix : list
        the last k bits

    Returns
    -------
    energies : np.array
        vector of energies such that E = energies.dot(amplitudes)
        where amplitudes are absolute values squared of qiskit statevector
    """
    if len(postfix) == 0:
        # Векторизованная версия для лучшей производительности
        bit_strings = generate_bit_strings(nbits)
        
        # Разбиваем на чунки для параллельной обработки
        chunk_size = max(1, len(bit_strings) // num_processes)
        chunks = [bit_strings[i:i + chunk_size] for i in range(0, len(bit_strings), chunk_size)]
        
        def process_chunk(chunk):
            return np.array([obj_f(x) for x in chunk], dtype=np.float64)
        
        with Pool(num_processes) as pool:
            results = pool.map(process_chunk, chunks)
        
        return np.concatenate(results)
    else:
        # Оригинальная реализация для случаев с postfix
        bit_strings = (np.hstack([x, postfix]) for x in yield_all_bitstrings(nbits - len(postfix)))
        
        with Pool(num_processes) as pool:
            ens = np.array(pool.map(obj_f, bit_strings), dtype=np.float64)
        return ens


def obj_from_statevector(sv, obj_f, precomputed_energies=None):
    """Compute objective from Qiskit statevector
    For large number of qubits, this is slow.
    """
    if precomputed_energies is not None:
        probabilities = np.abs(sv) ** 2
        return _obj_from_statevector_jit(probabilities, precomputed_energies)
    else:
        qubit_dims = np.log2(sv.shape[0])
        if qubit_dims % 1:
            raise ValueError("Input vector is not a valid statevector for qubits.")
        qubit_dims = int(qubit_dims)
        
        bit_strings = generate_bit_strings(qubit_dims)
        return sum(obj_f(bit_strings[kk]) * (np.abs(sv[kk]) ** 2) for kk in range(sv.shape[0]))


def objective_from_counts(counts, obj):
    """Compute expected value of the objective from shot counts"""
    if len(counts) > 100:  # JIT эффективен для больших словарей
        try:
            # Преобразуем словарь в массивы для JIT
            keys = []
            values = []
            
            for meas, meas_count in counts.items():
                key_array = np.array([int(x) for x in meas], dtype=np.int32)
                keys.append(key_array)
                values.append(meas_count)
            
            # Создаем JIT-совместимые массивы
            keys_array = np.array(keys)
            values_array = np.array(values, dtype=np.int64)
            
            # JIT-компилируем объективную функцию
            jit_obj = jit(nopython=True, cache=True, fastmath=True)(obj)
            
            return _objective_from_counts_jit(keys_array, values_array, jit_obj)
            
        except:
            pass
    
    # Fallback к оригинальной реализации
    mean = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = obj(np.array([int(x) for x in meas], dtype=np.int32))
        mean += obj_for_meas * meas_count
        total_counts += meas_count
    return mean / total_counts


def brute_force_optimization(obj_f, num_variables: int, minimize: bool = False, function_takes: str = "spins", *args: object, **kwargs: object):
    """Get the maximum of a function by complete enumeration
    Returns the maximum value and the extremizing bit string
    """
    bit_strings = generate_bit_strings(num_variables)
    
    # JIT-оптимизация для случаев без дополнительных аргументов
    if not args and not kwargs:
        try:
            jit_obj_f = jit(nopython=True, cache=True, fastmath=True)(obj_f)
            
            best_cost, best_idx = _brute_force_optimization_jit(
                bit_strings, jit_obj_f, minimize, function_takes == "spins"
            )
            
            return best_cost, bit_strings[best_idx]
            
        except:
            pass
    
    # Fallback к оптимизированной версии оригинальной реализации
    if minimize:
        best_cost_brute = np.inf
        compare = lambda x, y: x < y
    else:
        best_cost_brute = -np.inf
        compare = lambda x, y: x > y
    
    xbest_brute = bit_strings[0]
    
    for x in bit_strings:
        if function_takes == "spins":
            cost = obj_f(1 - 2 * x.astype(np.int32), *args, **kwargs)
        elif function_takes == "bits":
            cost = obj_f(x.astype(np.int32), *args, **kwargs)
        
        if compare(cost, best_cost_brute):
            best_cost_brute = cost
            xbest_brute = x
    
    return best_cost_brute, xbest_brute