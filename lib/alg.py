import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.quantum_info import SparsePauliOp, Operator, Statevector
from typing import Union
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit import Measure

from lib.utils import extract_N_and_K, qubitize_price_system, to_quantum_circuit

def get_probas(input):
    """
    Gets the probability distribution given by the Quantum Circuit or Statevector.
    """
    if isinstance(input, (QuantumCircuit, Statevector)):
        ansatzz = to_quantum_circuit(input)
        psi = Statevector.from_instruction(ansatzz)
    elif isinstance(input, list):  # Assuming input is a list of complex numbers
        input = [complex(x) for x in input]  # Ensure the input is a list of complex numbers
        psi = Statevector(input)
    else:
        raise ValueError("Input must be a QuantumCircuit, Statevector, or list of complex numbers")
    
    proba_vector = np.abs(psi.data) ** 2
    return proba_vector

def calculate_shots(K, delta=0.1, epsilon=0.1, min_p=None, max_shots=5000):
    """
    Calculate the number of shots needed based on Hoeffding's inequality,
    with an upper limit on the number of shots.
    
    Parameters:
    - K: Number of non-null probabilities of interest.
    - delta: Desired confidence level (1 - delta).
    - epsilon: Desired accuracy.
    - min_p: Minimum probability among the non-null terms (if known). If None, assumes 1/K.
    - max_shots: Maximum number of shots allowed (default is 10,000).
    
    Returns:
    - shots: The number of shots to be used in the sampling, capped at max_shots.
    """
    if min_p is None:
        min_p = 1 / K
    
    n_samples = np.log(K / delta) / (2 * epsilon**2 * min_p)
    shots = int(np.ceil(n_samples))
    
    # Apply the cap on the number of shots
    shots = min(shots, max_shots)
    
    return shots

def get_sampling_proba(ansatz, S, delta=0.01, epsilon=0.05, initial_shots=None):
    """
    Simulate the quantum circuit and return the probabilities of each measurement outcome.
    
    Parameters:
    - ansatz: The quantum circuit to be simulated.
    - S: The S matrix used to extract N and K.
    - delta: Desired confidence level (1 - delta).
    - epsilon: Desired accuracy.
    - initial_shots: Initial number of shots for adaptive sampling. If None, calculate using worst-case.
    
    Returns:
    - probas_sampled: The sampled probability distribution.
    - shots: The number of shots used in the sampling.
    """
    # Extract the number of qubits from the ansatz
    num_qubits = ansatz.num_qubits
    
    # Extract K from the S matrix using extract_N_and_K
    N, K = extract_N_and_K(S)
    
    # Calculate or use provided initial number of shots
    if initial_shots is None:
        initial_shots = calculate_shots(K, delta, epsilon)
    
    # Copy the original circuit
    circuit_with_measurement = ansatz.copy()
    
    # Add measurements to the circuit copy
    if not circuit_with_measurement.cregs:
        circuit_with_measurement.measure_all()
    
    # Use Aer's qasm_simulator
    simulator = Aer.get_backend('qasm_simulator')
    
    # Transpile the circuit for the simulator
    compiled_circuit = transpile(circuit_with_measurement, simulator)
    
    # Execute the circuit on the qasm simulator with the initial number of shots
    result = simulator.run(compiled_circuit, shots=initial_shots).result()
    
    # Get the counts
    counts = result.get_counts(compiled_circuit)
    
    # Calculate the probabilities from the counts
    total_counts = sum(counts.values())
    probas_sampled = np.zeros(2 ** num_qubits)
    for bitstring, count in counts.items():
        index = int(bitstring, 2)  # Convert bitstring to index
        probas_sampled[index] = count / total_counts
    
    # Adaptive step: find the smallest non-zero probability and refine the shot estimate
    non_zero_probs = probas_sampled[probas_sampled > 0]
    if len(non_zero_probs) > 0:
        min_observed_p = np.min(non_zero_probs)
        # Recalculate shots based on the observed minimum probability
        additional_shots = calculate_shots(K, delta, epsilon, min_p=min_observed_p)
        if additional_shots > initial_shots:
            # Run additional shots if the new estimate requires more samples
            result = simulator.run(compiled_circuit, shots=additional_shots - initial_shots).result()
            counts = result.get_counts(compiled_circuit)
            total_counts += sum(counts.values())
            for bitstring, count in counts.items():
                index = int(bitstring, 2)
                probas_sampled[index] += count / total_counts
    
    # Return the sampled probabilities and the total number of shots used
    return probas_sampled, total_counts

def generalized_cost_function(
    S: np.ndarray,
    Pi: np.ndarray,
    D: np.ndarray,
    i: int,
    ansatz: QuantumCircuit,
    gamma: float,
    lambda_=0.1,
):
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)

    probas = get_probas(ansatz)
    Sp = S @ probas
    denominator = (np.linalg.norm(Sp)) ** 2
    Pi = Pi.reshape(-1)

    numerator = np.abs(np.dot(Pi, Sp)) ** 2
    proba_lost = proba_loss(S, Pi, ansatz)

    if gamma != 0:
        Di = D[i]
        D = D.flatten()
        derivative_induced_cost = (Di @ probas) ** 2
        numerator += gamma * derivative_induced_cost

    overlap_term = 1 - (numerator / denominator)
    cost = overlap_term + lambda_ * proba_lost

    shots_used = 0  # No sampling used in this function

    return np.real(overlap_term), np.real(proba_lost), np.real(cost), shots_used

def generalized_cost_function_sampling(
    S: np.ndarray,
    Pi: np.ndarray,
    D: np.ndarray,
    i: int,
    ansatz: QuantumCircuit,
    gamma: float,
    lambda_=0.5,
):
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)

    probas_sampled, shots = get_sampling_proba(ansatz, S)  # Now also return shots
    Sp_sampled = S @ probas_sampled
    Pi = Pi.reshape(-1)
    numerator = np.abs(np.dot(Pi, Sp_sampled)) ** 2
    denominator = (np.linalg.norm(Sp_sampled)) ** 2

    N, K = extract_N_and_K(S)
    proba_lost = 1 - probas_sampled[N: N + K].sum()

    if gamma != 0:
        Di = D[i] / norm_Pi
        D = D.flatten()
        derivative_induced_cost = (Di @ probas_sampled) ** 2
        numerator += gamma * derivative_induced_cost

    overlap_term = 1 - (numerator / denominator)
    cost = overlap_term + lambda_ * proba_lost

    return np.real(overlap_term), np.real(proba_lost), np.real(cost), shots  # Return shots used


def vqe_cost_function(
    S: np.ndarray, Pi: np.ndarray, D: np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function(S, Pi, D, i, ansatz, gamma=0, lambda_=lambda_)

def vqe_sampling_cost_function(
    S: np.ndarray, Pi: np.ndarray, D: np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function_sampling(S, Pi, D, i, ansatz, gamma=0, lambda_=lambda_)


def vqe_cost_function_for_dmin(
    S: np.ndarray, Pi: np.ndarray, D :np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function(S, Pi, D, i, ansatz, gamma=-1, lambda_=lambda_)

def vqe_cost_function_for_dmax(
    S: np.ndarray, Pi: np.ndarray, D: np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function(S, Pi, D, i, ansatz, gamma=1, lambda_=lambda_)

def vqe_sampling_cost_function_for_dmin(
    S: np.ndarray, Pi: np.ndarray, D :np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function_sampling(S, Pi, D, i, ansatz, gamma=-1, lambda_=lambda_)

def vqe_sampling_cost_function_for_dmax(
    S: np.ndarray, Pi: np.ndarray, D: np.ndarray, i: int, ansatz: QuantumCircuit, lambda_=1
):
    return generalized_cost_function_sampling(S, Pi, D, i, ansatz, gamma=1, lambda_=lambda_)

def normalized_overlap(q1, q2):
    """
    overlap between any two (non-normalized) vectors
    """
    if len(q1) != len(q2):
        raise ValueError("Vectors lengths do not match")
    inner_product = np.dot(q1.T, q2) / (np.linalg.norm(q1) * np.linalg.norm(q2))
    return np.abs(inner_product) ** 2


def overlap_with_Pi(S: np.ndarray, Pi: np.ndarray, ansatz: QuantumCircuit):
    """
    Overlap between S*p(\theta) and Pi
    """
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)
    current_probas = get_probas(ansatz)
    Sp = S @ current_probas
    return normalized_overlap(Sp, Pi)

def constrained_overlap_with_Pi(S: np.ndarray, Pi: np.ndarray, ansatz: QuantumCircuit):
    """
    Overlap between S*p(\theta) and Pi
    """
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)

    N_plus_one, K = extract_N_and_K(S)
    current_probas = get_probas(ansatz)
    relevant_probas = np.zeros_like(current_probas)
    relevant_probas[N_plus_one : N_plus_one + K] = current_probas[
        N_plus_one : N_plus_one + K
    ]
    Sp = S @ relevant_probas
    return normalized_overlap(Pi, Sp)

def constrained_distance_with_Pi(S: np.ndarray, Pi: np.ndarray, ansatz: QuantumCircuit):
    """
    Distance between S*p(\theta) and Pi
    """
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)

    N, K = extract_N_and_K(S)
    current_probas = get_probas(ansatz)
    relevant_probas = np.zeros_like(current_probas)
    relevant_probas[N : N + K] = current_probas[N : N + K]

    Sp = norm_Pi * (S @ relevant_probas)
    # Sp = S @ relevant_probas
    vec_Pi = norm_Pi*Pi.reshape(1, -1)
    distance = ((np.linalg.norm(Sp - vec_Pi)) ) / (norm_Pi)
    return distance

def proba_loss(S: np.ndarray, Pi: np.ndarray, ansatz: Union[QuantumCircuit, Statevector]):
    """
    The sum of probas in the range(N+1,N+K+1), ie the probas that should actually correspond to the martingale measure
    """
    if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
        S, Pi, norm_Pi = qubitize_price_system(S, Pi)
    else:
        _, _, norm_Pi = qubitize_price_system(S, Pi)

    N, K = extract_N_and_K(S)
    current_probas = get_probas(ansatz)
    return 1 - current_probas[N : N + K].sum()

def projector_on_zero(n, i):
    """
    /Helper for local_cost_function_exact./
    Create the projector |0><0| for the i-th qubit in an n-qubit system.
    First qubit is i=0, last qubit is i=n-1
    """

    # Projector |0><0| for a single qubit
    projector = np.array([[1, 0], [0, 0]])

    # Construct the full projector using Kronecker products
    if i == 0:
        return np.kron(projector, np.eye(2 ** (n - 1)))
    elif i == n - 1:
        return np.kron(np.eye(2 ** (n - 1)), projector)
    else:
        return np.kron(np.kron(np.eye(2 ** (i - 1)), projector), np.eye(2 ** (n - i)))


def hadamard_n_qubits(n):
    """
    /Helper for local_cost_function_exact./
    Construct the unitary matrix U_pi such that U_pi |0> = |Pi>, where |Pi> is a vector of ones up to normalization.
    This is achieved using a tensor product of Hadamard gates.
    """
    # Tensor product of n Hadamard gates
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    Hn = H
    for _ in range(n - 1):
        Hn = np.kron(Hn, H)

    return Hn

# def hadamard_test(
#     U: Union[SparsePauliOp, Operator, Gate, QuantumCircuit]
# , shots=5000
# ):
#     """
#     Returns <0| U |0> using the Hadamard test
#     Args:
#         U (SparsePauliOp | Operator | Gate): the unitary in the Hadamard test
#         shots (int): the number of shots in the simulation
#     Returns:
#         complex: the estimated value <0| U |0>
#     """
#     n = U.num_qubits
#     U_circ = QuantumCircuit(n)
#     if isinstance(U, Operator) or isinstance(U, SparsePauliOp):
#         U_circ.append(U.to_operator(), range(n))  # transform U into a circuit
#     elif isinstance(U, Gate):
#         U_circ.append(U, range(n))
#     elif isinstance(U, QuantumCircuit):
#         U_circ = U

#     # get the real part
#     qc = QuantumCircuit(n + 1, 1)
#     qc.h(0)
#     qc.append(U_circ.control(), range(n + 1))  # controlled U;
#     # be careful about the annoying qubit ordering in qiskit, e.g., XZ applies Z first and then X
#     qc.h(0)
#     qc.measure(0, 0)
#     qc = transpile(qc, sim)
#     # print(qc)

#     result = sim.run(qc, shots=shots).result()
#     real_part = 0
#     for key in result.get_counts():
#         real_part += (-1) ** int(key) * result.get_counts()[key] / shots

#     # get the imaginary part
#     qc = QuantumCircuit(n + 1, 1)
#     qc.h(0)
#     qc.sdg(0)
#     qc.append(U_circ.control(), range(n + 1))  # controlled U
#     qc.h(0)
#     qc.measure(0, 0)
#     qc = transpile(qc, sim)
#     # print(qc)

#     result = sim.run(qc, shots=shots).result()
#     imaginary_part = 0
#     for key in result.get_counts():
#         imaginary_part += (-1) ** int(key) * result.get_counts()[key] / shots

#     # print(real_part, imaginary_part)

#     return real_part + 1j * imaginary_part


# def cost_function(
#     S: np.ndarray, Pi: np.ndarray, ansatz: QuantumCircuit, shots=5000, sim=sim
# ):
#     """
#     Return $| E_{j \sim \ket{\psi(\theta)}} \mel{\Pi}{S}{j} |$.
#     """
#     if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
#         S, Pi, norm_Pi = qubitize_price_system(S, Pi)
#     else:
#         _, _, norm_Pi = qubitize_price_system(S, Pi)
#     n = int(np.log2(S.shape[0]))
#     S = pauli_decompose(S)

#     U_Pi = QuantumCircuit(n)
#     Pi = Pi.flatten()
#     U_Pi.prepare_state(Pi / np.linalg.norm(Pi))

#     # sample from ansatz state
#     ansatz = transpile(ansatz, sim)
#     result = sim.run(ansatz, shots=shots).result()
#     exp_res = result.get_counts()

#     overlap = 0
#     for bitstring in exp_res:
#         overlap_j = (
#             0  # \sum_l a_l <0| U_Pi^\dagger P_l X(a) |0>, where a is the bit string
#         )
#         for l, pauli in enumerate(S.paulis):
#             qc = QuantumCircuit(n)
#             for i, x_i in enumerate(bitstring):
#                 qc.x(i) if x_i == "1" else None
#             qc.append(pauli.to_instruction(), range(n))
#             qc.compose(U_Pi.inverse(), inplace=True)

#             overlap_j += hadamard_test(qc, shots=shots, sim=sim) * S.coeffs[l]
#         overlap += overlap_j * exp_res[bitstring] / shots

#     return np.abs(overlap)

# def local_cost_function_penalty(
#     S: np.ndarray, Pi: np.ndarray, ansatz: QuantumCircuit, lambda_=5
# ):
#     if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
#         S, Pi, norm_Pi = qubitize_price_system(S, Pi)
#     else:
#         _, _, norm_Pi = qubitize_price_system(S, Pi)

#     n_qubits = ansatz.num_qubits
#     probas = get_probas(ansatz)

#     # Precompute projectors and U_pi
#     projectors = [projector_on_zero(n_qubits, i) for i in range(n_qubits)]
#     U_pi = hadamard_n_qubits(n_qubits)
#     U_pi_dagger = U_pi.T.conj()

#     numerator = 0
#     denominator = 0

#     # Precompute full_ket and full_bra arrays
#     full_kets = np.dot(U_pi_dagger, S)
#     full_bras = np.dot(S.T.conj(), U_pi)

#     for j in range(S.shape[1]):
#         full_ket = full_kets[:, j]
#         for j_prime in range(S.shape[1]):
#             full_bra = full_bras[j_prime, :]

#             # Calculate denominator terms once
#             denominator += (
#                 probas[j] * probas[j_prime] * np.dot(S[:, j_prime].T.conj(), S[:, j])
#             )

#             # Calculate numerator terms
#             expectation_i = 0
#             for projector_i in projectors:
#                 full_term = np.dot(full_bra, np.dot(projector_i, full_ket))
#                 expectation_i += probas[j] * probas[j_prime] * full_term

#             numerator += expectation_i

#     proba_lost = proba_loss(S, Pi, ansatz)
#     penalty = lambda_ * proba_lost

#     cost = 1 - (numerator / denominator) + penalty
#     return np.real(cost)


# def global_quantum_cost_function_penalty(
#     S: np.ndarray,
#     Pi: np.ndarray,
#     ansatz: QuantumCircuit,
#     lambda_=1,
#     num_samples=1000,
#     shots=5000,
#     sim=None,
# ):
#     if sim is None:
#         sim = AerSimulator()
#     if S.shape[0] != S.shape[1] or np.log2(S.shape[0]) % 1 != 0:
#         S, Pi, norm_Pi = qubitize_price_system(S, Pi)
#     else:
#         _, _, norm_Pi = qubitize_price_system(S, Pi)

#     qc = transpile(ansatz, sim)
#     # print(qc)

#     result = sim.run(qc, shots=shots).result()
#     counts = result.get_counts()
#     sampled_indices_j = []
#     for bitstring, count in counts.items():
#         index = int(bitstring, 2)
#         sampled_indices_j.extend([index] * count)
#     sampled_indices_j = np.random.choice(sampled_indices_j, size=num_samples)

#     N_plus_one, K = extract_N_and_K(S)
#     result_func = 0
#     denominator = 0
#     pauli_decomposition = pauli_decompose(S)
#     n_qubits = ansatz.num_qubits

#     for j, jp in zip(sampled_indices_j, sampled_indices_jp):
#         j_str = format(j, f"0{n_qubits}b")
#         jp_str = format(jp, f"0{n_qubits}b")

#         X_j = create_state_preparation_circuit(n_qubits, j_str)
#         X_jp = create_state_preparation_circuit(n_qubits, jp_str)

#         for l in range(len(pauli_decomposition)):
#             c_l, S_l = pauli_decomposition[l]
#             U_pi_dagger_S_l_X_j = BaseOperator(X_j).compose(S_l).compose(U_pi_dagger)
#             term_1 = hadamard_test(U_pi_dagger_S_l_X_j, shots=shots, sim=sim)
#             result += c_l * term_1

#             for lp in range(len(pauli_decomposition)):
#                 c_lp, S_lp = pauli_decomposition[lp]
#                 U_dagger_U = (
#                     BaseOperator(X_jp).compose(S_lp.T.conj()).compose(S_l).compose(X_j)
#                 )
#                 term_2 = hadamard_test(U_dagger_U, shots=shots, sim=sim)
#                 denominator += c_lp * c_l * term_2