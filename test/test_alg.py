import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import HGate, UGate
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from qiskit_aer import AerSimulator

from lib.alg import (
    hadamard_test,
    cost_function,
    global_cost_function_exact,
    get_probas,
    local_cost_function_penalty,
)


def test_hadamard_test():
    A = SparsePauliOp(["XZ"])
    assert hadamard_test(A) == pytest.approx(0, abs=0.05)

    A = SparsePauliOp(["ZZ"])
    assert hadamard_test(A) == pytest.approx(1, abs=0.05)

    A = HGate()
    assert hadamard_test(A) == pytest.approx(1 / np.sqrt(2), abs=0.05)

    A = UGate(0.1, 0.2, 0.3)
    assert hadamard_test(A) == pytest.approx(A.to_matrix()[0, 0], abs=0.05)

    A = random_circuit(3, 3)
    assert hadamard_test(A) == pytest.approx(Operator(A).to_matrix()[0, 0], abs=0.05)


def test_cost_function():
    S = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
    Pi = (1 / np.sqrt(2)) * np.array([[1], [1]])

    ansatz = QuantumCircuit(1)
    ansatz.h(0)
    ansatz.measure_all()

    assert cost_function(S, Pi, ansatz) == pytest.approx(0.5, abs=0.05)


# from qiskit_aer import AerSimulator

# sim = AerSimulator(method = "statevector")

# qc1 = QuantumCircuit(1)
# qc1.prepare_state([1/np.sqrt(2)]*2)
# qc1.x(0)

# qc2 = QuantumCircuit(1)
# qc2.h(0)
# qc2.x(0)

# # for instruction in qc1[::-1]:
# #     qc2.append(instruction)
# qc2.compose(qc1.inverse(), inplace = True)
# print(qc2)

# qc2.save_statevector()

# qc2 = transpile(qc2, sim)

# print(qc2)

# result = sim.run(qc2).result()
# print(result.get_statevector())

# qc1 = QuantumCircuit(1)
# qc1.prepare_state([1/np.sqrt(2)]*2)
# # qc1.
# qc1.save_statevector()
# qc1 = transpile(qc1, sim)

# print(qc1)

# result = sim.run(qc1).result()
# print(result.get_statevector())


def test_get_probas():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.h(1)
    probas = get_probas(qc)
    assert np.isrealobj(probas)


def test_global_cost_function_exact():
    # sim = AerSimulator(method = "statevector")

    qc1 = QuantumCircuit(1)
    qc1.h(0)
    psi = Statevector.from_instruction(qc1)
    # probas = np.abs(psi.data)**2
    S = np.random.rand(2, 2)
    Pi = np.array([1, 1])

    assert global_cost_function_exact(S, Pi, qc1) == pytest.approx(
        0.5 * np.sum(S), abs=0.05
    )
