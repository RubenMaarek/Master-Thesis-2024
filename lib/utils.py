import numpy as np
from qiskit.quantum_info import SparsePauliOp, Operator
import matplotlib.pyplot as plt
from scipy.linalg import lstsq
from scipy.optimize import nnls, minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.quantum_info import Statevector
from typing import Union
from cvxopt import matrix, solvers
from pulp import *

def to_quantum_circuit(
    U: Union[SparsePauliOp, Operator, Gate, QuantumCircuit,Statevector],
):
    """
    Converts the input to a QuantumCircuit object.

    Parameters:
    - U (SparsePauliOp | Operator | Gate | QuantumCircuit): the unitary to convert.

    Returns:
    - QuantumCircuit: The converted quantum circuit.
    """
    if isinstance(U, QuantumCircuit):
        return U
    elif isinstance(U, Gate):
        n = U.num_qubits
        qc = QuantumCircuit(n)
        qc.append(U, range(n))
        return qc
    elif isinstance(U, Operator) or isinstance(U, SparsePauliOp):
        n = U.num_qubits
        qc = QuantumCircuit(n)
        qc.append(U.to_instruction(), range(n))
        return qc
    elif isinstance(U, Statevector):
        n = U.num_qubits
        qc = QuantumCircuit(n)
        qc.initialize(U.data, range(n))
        return qc
    else:
        raise TypeError("Unsupported type for conversion to QuantumCircuit.")
    
def make_hermitian(A):
    """
    Given a matrix A, return a Hermitian matrix
    0 & A \\
    A^dagger & 0
    """
    m, n = A.shape
    A_herm = np.zeros((m + n, m + n), dtype=complex)
    A_herm[:m, m:] = A
    A_herm[m:, :m] = A.conj().T

    return A_herm


def pad_zeros(A):
    """
    Given a matrix A, return a matrix of size 2^k x 2^k with A in the top-left corner
    """
    m, n = A.shape
    k = np.ceil(np.log2(max(m, n))).astype(int)
    A_padded = np.zeros((2**k, 2**k), dtype=complex)
    A_padded[:m, :n] = A

    return A_padded


def qubitize_price_system(S: np.ndarray, Pi: np.ndarray, D : np.ndarray = None):
    """
    Make the dimensions of S and Pi compatible with the 2^n.
    Args:
        S (np.ndarray): the matrix S
        Pi (np.ndarray): the vector Pi. Note that its shape is assumed to be (N+1, 1)
    """
    norm_Pi = np.linalg.norm(Pi)
    Pi = Pi.reshape(-1, 1)
    Sh = make_hermitian(S)
    Sz = pad_zeros(Sh)
    N = Sz.shape[0]
    Pi = np.concatenate((Pi, np.zeros((N - len(Pi), 1))), axis=0)
    Sz /= norm_Pi
    Pi /= norm_Pi
    return np.real(Sz), np.real(Pi), np.real(norm_Pi)


def pauli_decompose(A: Union[Operator , np.ndarray]):
    """
    Given an Operator A, return its Pauli decomposition. Wrapper around SparsePauliOp.from_operator
    """
    if isinstance(A, np.ndarray):
        A = Operator(A)

    A = SparsePauliOp.from_operator(A)

    return A


def plot(cost_fctn, title, label, xlabel="Iteration", ylabel="Loss", figsize=(10, 5)):
    plt.figure(figsize=figsize)
    plt.plot(cost_fctn, "b-", label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def extract_N_and_K(A_padded):
    """
    Extract the number of rows (N) and columns (K) from the padded matrix A_padded.
    Args:
        A_padded (np.ndarray): The padded Hermitian matrix.
    Returns:
        int, int: The number of rows (N) and the number of columns (K) of the original matrix A.
    """
    # Find num_rows (N+1) by locating the last non-zero element in the first row
    first_row = A_padded[0]
    num_rows = np.where(first_row != 0)[0][0]

    # The original S starts at (0, num_rows)
    start_col = num_rows

    # Extract the columns of S until encountering a null column
    end_col = start_col
    while end_col < A_padded.shape[1] and not np.all(A_padded[:, end_col] == 0):
        end_col += 1

    # Determine the number of columns in the original S matrix (num_cols)
    num_cols = end_col - start_col

    return num_rows, num_cols


def extract_original_matrix(A_tilde, N, K):
    """
    Extracts the original matrix S from the given S_tilde.

    Args:
        A_tilde (np.ndarray): The 2^n matrix containing S and its conjugate transpose S†.
        N (int): The number of rows of the original matrix S.
        K (int): The number of columns of the original matrix S.

    Returns:
        np.ndarray: The original N x K matrix S.
    """
    # Ensure S_tilde is a numpy array
    A_tilde = np.asarray(A_tilde)
    
    # Extract the original S matrix from the top right block of S_tilde
    A = A_tilde[:N, N:N+K]
    
    return A



def qp_solution(A: np.ndarray, b: np.ndarray):
    """
    Solve the quadratic programming problem min_x ||Ax - b||^2 subject to x >= 0 and sum x_i = 1.
    Translate it to min (1/2)x.T Q x + q.T x s.t. Gx <= h, Rx = s.
    """
    P = 2* A.T @ A 
    q = -2 * A.T @ b

    G = -np.eye(A.shape[1])
    h = np.zeros(A.shape[1])

    R = np.ones((1, A.shape[1]))
    s = np.array([1.0])

    P,q,G,h,R,s = matrix(P), matrix(q), matrix(G), matrix(h), matrix(R), matrix(s)

    # Solve the QP problem
    sol = solvers.qp(P, q, G, h, R, s)
    x = np.array(sol['x']).flatten()

    return x

def solve_lp_problem(D, i, S, Pi):

    N, K = extract_N_and_K(S)
    original_D, original_S, original_Pi = extract_original_matrix(D,N,K), extract_original_matrix(S,N,K), Pi[:N]
        
    results = {}
    
    for problem_type in ['min', 'max']:
        # Create the LP problem
        prob = LpProblem(f"{problem_type.capitalize()}_Di_q", LpMinimize if problem_type == 'min' else LpMaximize)

        # Define decision variables
        q = [LpVariable(f"q{j}", lowBound=0) for j in range(K)]

        # Set objective function
        prob += lpSum([D[i,j] * q[j] for j in range(K)])  # D[i] @ q

        # Add constraints
        prob += lpSum(q) == 1  # Sum of qi = 1
        prob += lpSum([S[j] * q[j] for j in range(K)]) == Pi  # Sq = Pi

        # Solve the problem
        prob.solve(PULP_CBC_CMD(msg=0))  # msg=0 suppresses solver output

        # Store the result
        results[problem_type] = value(prob.objective)

    return results['min'], results['max']