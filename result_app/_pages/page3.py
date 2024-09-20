import streamlit as st
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from qiskit.quantum_info import Statevector
from pulp import * 
from cvxopt import matrix, solvers

def get_probas_from_csv(csv_series):
    complex_list = [complex(num) for num in csv_series]
    psi = Statevector(complex_list)
    return np.abs(psi.data) ** 2

def load_data(selected_run_dir):
    selected_run_path = Path(selected_run_dir)
    
    with open(selected_run_path / "model_config.json", 'r') as json_file:
        model_config = json.load(json_file)
    
    all_option_prices_df = pd.read_csv(selected_run_path / "all_option_prices.csv")
    D_matrix = pd.read_csv(selected_run_path / "D_matrix.csv").values
    final_statevectors_df = pd.read_csv(selected_run_path / "final_statevectors.csv")
    final_statevectors_lp_df = pd.read_csv(selected_run_path / "final_statevectors_lp.csv")
    S_matrix = pd.read_csv(selected_run_path / "S_matrix.csv").values
    Pi_vector = pd.read_csv(selected_run_path / "Pi_vector.csv", header=None).values.flatten()
    # qp_vector = pd.read_csv(selected_run_path / "qp_solution_vector.csv", header=None).values.flatten() 

    return model_config, all_option_prices_df, D_matrix, S_matrix, Pi_vector, final_statevectors_df, final_statevectors_lp_df

def calculate_option_prices(D_matrix, final_statevectors_df, N):
    vqe_probas = get_probas_from_csv(final_statevectors_df['vqe_cost_function'])
    vqe_sampling_probas = get_probas_from_csv(final_statevectors_df['vqe_sampling_cost_function'])

    vqe_option_prices = (D_matrix @ vqe_probas)[:N]
    vqe_sampling_option_prices = (D_matrix @ vqe_sampling_probas)[:N]

    return vqe_option_prices, vqe_sampling_option_prices

def calculate_ratios(all_option_prices_df, vqe_option_prices, vqe_sampling_option_prices):
    all_option_prices_df['VQE_Option_Prices'] = vqe_option_prices
    all_option_prices_df['VQE_Sampling_Option_Prices'] = vqe_sampling_option_prices

    vqe_mc_ratios = all_option_prices_df['VQE_Option_Prices'] - all_option_prices_df['MC_Option_Prices']
    vqe_sampling_mc_ratios = all_option_prices_df['VQE_Sampling_Option_Prices'] - all_option_prices_df['MC_Option_Prices']
    vqe_qp_ratios = all_option_prices_df['VQE_Option_Prices'] - all_option_prices_df['QP_Option_Prices']
    vqe_sampling_qp_ratios = all_option_prices_df['VQE_Sampling_Option_Prices'] - all_option_prices_df['QP_Option_Prices']

    return vqe_mc_ratios, vqe_sampling_mc_ratios, vqe_qp_ratios, vqe_sampling_qp_ratios

def plot_ratios(all_option_prices_df, vqe_mc_ratios, vqe_sampling_mc_ratios, vqe_qp_ratios, vqe_sampling_qp_ratios):
    indices = np.arange(len(all_option_prices_df))
    bar_width = 0.35

    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(indices - bar_width/2, vqe_mc_ratios, bar_width, label='VQE-MC', color='green')
    ax1.bar(indices + bar_width/2, vqe_sampling_mc_ratios, bar_width, label='VQE Sampling-MC', color='red')
    ax1.axhline(y=0, color='blue', linestyle='--', label='MC Benchmark Line')
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Difference")
    ax1.set_title("VQE Prices Relative to MC Prices")
    ax1.set_xticks(indices)
    ax1.set_xticklabels(all_option_prices_df['Index'])
    ax1.legend()
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.bar(indices - bar_width/2, vqe_qp_ratios, bar_width, label='VQE-QP', color='green')
    ax2.bar(indices + bar_width/2, vqe_sampling_qp_ratios, bar_width, label='VQE Sampling-QP', color='red')
    ax2.axhline(y=0, color='orange', linestyle='--', label='QP Benchmark Line')
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Difference")
    ax2.set_title("VQE Prices Relative to QP Prices")
    ax2.set_xticks(indices)
    ax2.set_xticklabels(all_option_prices_df['Index'])
    ax2.legend()
    st.pyplot(fig2)

def calculate_rmse(all_option_prices_df, vqe_option_prices, vqe_sampling_option_prices):
    rmse_vqe_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], vqe_option_prices))
    rmse_vqe_sampling_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], vqe_sampling_option_prices))
    rmse_vqe_qp = np.sqrt(mean_squared_error(all_option_prices_df['QP_Option_Prices'], vqe_option_prices))
    rmse_vqe_sampling_qp = np.sqrt(mean_squared_error(all_option_prices_df['QP_Option_Prices'], vqe_sampling_option_prices))
    rmse_qp_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], all_option_prices_df['QP_Option_Prices']))

    norm_rmse_vqe_mc = rmse_vqe_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100
    norm_rmse_vqe_sampling_mc = rmse_vqe_sampling_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100
    norm_rmse_qp_mc = rmse_qp_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100

    return {
        "VQE vs MC": f"{norm_rmse_vqe_mc:.2f}%",
        "VQE Sampling vs MC": f"{norm_rmse_vqe_sampling_mc:.2f}%",
        "QP vs MC": f"{norm_rmse_qp_mc:.2f}%"
    }

def calculate_lp_prices(D_matrix, final_statevectors_lp_df, i):
    d_min = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_cost_function_for_dmin'])
    d_min_sampling = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_sampling_cost_function_for_dmin'])
    d_max = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_cost_function_for_dmax'])
    d_max_sampling = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_sampling_cost_function_for_dmax'])
    return d_min, d_min_sampling, d_max, d_max_sampling

def plot_lp_prices(d_min, d_max, vqe_price, qp_price, lp_min, lp_max, title, method):
    # Set up data for the grouped bar chart
    categories = ['Min', 'Price', 'Max']
    quantum_values = [d_min, vqe_price, d_max]
    classical_values = [lp_min, qp_price, lp_max]

    x = np.arange(len(categories))  # the label locations
    bar_width = 0.35  # Width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot Quantum results
    quantum_bars = ax.bar(x - bar_width/2, quantum_values, bar_width, label='Quantum (VQE)', color='green')

    # Plot Classical results
    classical_bars = ax.bar(x + bar_width/2, classical_values, bar_width, label='Classical (QP/LP)', color='purple')

    # Adding labels
    ax.set_ylabel("Option Price")
    ax.set_title(f"{title} Comparison - {method.capitalize()} Method")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    # Annotate the bars with their values
    for bars in [quantum_bars, classical_bars]:
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f'{yval:.2f}',
                ha='center',
                va='bottom' if yval < 1 else 'top',
                fontsize=10,
                color='black'
            )

    # Add a legend
    ax.legend()

    # Add grid lines for clarity
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Display the plot
    plt.xticks(rotation=45)
    st.pyplot(fig)

def main(selected_run_dir):
    st.title("Derivative Prices")

    # Load QP solution vector
    selected_run_path = Path(selected_run_dir)
    qp_solution_path = selected_run_path / "qp_solution_vector.csv"
    qp_vector = pd.read_csv(qp_solution_path, header=None).values.flatten()

    model_config, all_option_prices_df, D_matrix, S_matrix, Pi_vector, final_statevectors_df, final_statevectors_lp_df = load_data(selected_run_dir)
    N, i = model_config["N"], model_config["i"]

    vqe_option_prices, vqe_sampling_option_prices = calculate_option_prices(D_matrix, final_statevectors_df, N)
    vqe_mc_ratios, vqe_sampling_mc_ratios, vqe_qp_ratios, vqe_sampling_qp_ratios = calculate_ratios(all_option_prices_df, vqe_option_prices, vqe_sampling_option_prices)

    plot_ratios(all_option_prices_df, vqe_mc_ratios, vqe_sampling_mc_ratios, vqe_qp_ratios, vqe_sampling_qp_ratios)

    rmse_metrics = calculate_rmse(all_option_prices_df, vqe_option_prices, vqe_sampling_option_prices)
    st.header("Normalized RMSE (as %)")
    st.table(pd.DataFrame(rmse_metrics, index=["Normalized RMSE"]).T)

    st.write(f"i: {i}")
    d_min, d_min_sampling, d_max, d_max_sampling = calculate_lp_prices(D_matrix, final_statevectors_lp_df, i)
    qp_price_i = all_option_prices_df['QP_Option_Prices'][i]
    lp_min, lp_max = solve_lp_problem(D_matrix, i, S_matrix, Pi_vector)

    if all(value == 0.0 for value in [d_min, d_min_sampling, d_max, d_max_sampling]):
        st.warning("All calculated option prices (d_min, d_max) are zero. Skipping the plot to avoid rendering issues.")
    else:
        st.write(f"d_min: {d_min}, d_min_sampling: {d_min_sampling}, d_max: {d_max}, d_max_sampling: {d_max_sampling}")
        st.write(f"QP price: {qp_price_i}, LP min: {lp_min}, LP max: {lp_max}")
        plot_lp_prices(d_min, d_max, vqe_option_prices[i], qp_price_i, lp_min, lp_max, "LP Pricing Method", "Exact")
        plot_lp_prices(d_min_sampling, d_max_sampling, vqe_sampling_option_prices[i], qp_price_i, lp_min, lp_max, "LP Pricing Method", "Sampling")

if __name__ == "__main__":
    main("selected_run_dir")  # Replace with actual directory selection logic



# Imported Utils functions : 
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