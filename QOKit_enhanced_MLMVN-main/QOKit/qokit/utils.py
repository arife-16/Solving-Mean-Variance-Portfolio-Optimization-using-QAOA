###############################################################################
# // SPDX-License-Identifier: Apache-2.0
# // Copyright : JP Morgan Chase & Co
###############################################################################
import numpy as np
import pandas as pd
import qiskit
from qiskit import QuantumCircuit
from pytket import OpType
from pytket.circuit import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.qiskit import qiskit_to_tk
from importlib_resources import files
from qiskit.providers.basic_provider import BasicProvider

from pytket.passes import (
    SequencePass,
    AutoSquash,
    RemoveRedundancies,
    SimplifyInitial,
    FullPeepholeOptimise,
    NormaliseTK2,
    DecomposeTK2,
    CommuteThroughMultis,
)


from qokit.energy_utils import (
    precompute_energies, 
    precompute_energies_parallel, 
    obj_from_statevector, 
    objective_from_counts,
    brute_force_optimization
)






def get_all_best_known():
    """Get both dataframes of best known parameters for LABS merged into one

    Returns
    ----------
    df : pandas.DataFrame
        Columns corresponding to results where the parameters were optimized
        with respect to merit factor have ' opt4MF' appended to them,
        and columns optimized for overlap have ' opt4overlap' appended to them
    """
    df1 = pd.read_json(
        files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_MF.json"),
        orient="index",
    ).drop("nseeds", axis=1)
    df2 = pd.read_json(
        files("qokit.assets").joinpath("best_LABS_QAOA_parameters_wrt_overlap.json"),
        orient="index",
    )

    df1 = df1.set_index(["N", "p"]).add_suffix(" opt4MF")
    df2 = df2.set_index(["N", "p"]).add_suffix(" opt4overlap")

    return df1.merge(df2, left_index=True, right_index=True, how="outer").reset_index()


def brute_force(obj_f, num_variables: int, minimize: bool = False, function_takes: str = "spins", *args: object, **kwargs: object):
    """Get the maximum of a function by complete enumeration
    Returns the maximum value and the extremizing bit string
    
    This is a wrapper around brute_force_optimization from energy_utils for backward compatibility
    """
    return brute_force_optimization(obj_f, num_variables, minimize, function_takes, *args, **kwargs)


def reverse_array_index_bit_order(arr):
    arr = np.array(arr)
    n = int(np.log2(len(arr)))  # Calculate the value of N
    if n % 1:
        raise ValueError("Input vector has to have length 2**N where N is integer")

    index_arr = np.arange(len(arr))
    new_index_arr = np.zeros_like(index_arr)
    while n > 0:
        last_8 = np.unpackbits(index_arr.astype(np.uint8), axis=0, bitorder="little")
        repacked_first_8 = np.packbits(last_8).astype(np.int64)
        if n < 8:
            new_index_arr += repacked_first_8 >> (8 - n)
        else:
            new_index_arr += repacked_first_8 << (n - 8)
        index_arr = index_arr >> 8
        n -= 8
    return arr[new_index_arr]


def state_to_ampl_counts(vec, eps: float = 1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = "0{}b".format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2 + val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def unitary_from_circuit(qc: QuantumCircuit):
    backend = BasicProvider().get_backend("unitary_simulator")
    job = qiskit.transpile(qc, backend)
    U = job.result().get_unitary()
    return U


def get_ramp(delta, p: int):
    gamma = np.array([-delta * j / (p + 1) for j in range(1, p + 1)])
    beta = np.array([delta * (1 - j / (p + 1)) for j in range(1, p + 1)])
    return {"beta": beta, "gamma": gamma}


def transpile_hseries(quantinuum_backend: QuantinuumBackend, circuit: Circuit, num_passes_repeats: int = 2):
    """
    Transpile circuit to quantinuum backend
    circuit is qiskit.QuantumCircuit or pytket.circuit.Circuit
    """
    assert isinstance(quantinuum_backend, QuantinuumBackend)

    if isinstance(circuit, QuantumCircuit):
        circ = qiskit_to_tk(circuit)
    else:
        assert isinstance(circuit, Circuit)
        circ = circuit

    compiled_circuit = circ.copy()

    squash = auto_squash_pass({OpType.PhasedX, OpType.Rz})

    fidelities = {
        "ZZMax_fidelity": 0.999,
        "ZZPhase_fidelity": lambda x: 1.0 if not np.isclose(x, 0.5) else 0.9,
    }
    _xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])
    _xcirc.add_phase(0.5)
    passes = [
        RemoveRedundancies(),
        CommuteThroughMultis(),
        FullPeepholeOptimise(target_2qb_gate=OpType.TK2),
        NormaliseTK2(),
        DecomposeTK2(**fidelities),
        quantinuum_backend.rebase_pass(),
        squash,
        SimplifyInitial(allow_classical=False, create_all_qubits=True, xcirc=_xcirc),
    ]

    seqpass = SequencePass(passes)

    # repeat compilation steps `num_passes_repeats` times
    for _ in range(num_passes_repeats):
        seqpass.apply(compiled_circuit)

    ZZMax_depth = compiled_circuit.depth_by_type(OpType.ZZMax)
    ZZPhase_depth = compiled_circuit.depth_by_type(OpType.ZZPhase)
    ZZMax_count = len(compiled_circuit.ops_of_type(OpType.ZZMax))
    ZZPhase_count = len(compiled_circuit.ops_of_type(OpType.ZZPhase))
    two_q_depth = ZZMax_depth + ZZPhase_depth
    two_q_count = ZZMax_count + ZZPhase_count

    return compiled_circuit, {
        "two_q_count": two_q_count,
        "two_q_depth": two_q_depth,
        "ZZMAX": ZZMax_count,
        "ZZPhase": ZZPhase_count,
    }


def invert_counts(counts):
    """Convert from lsb to msb ordering and vice versa"""
    return {k[::-1]: v for k, v in counts.items()}