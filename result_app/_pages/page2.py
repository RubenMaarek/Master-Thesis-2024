import streamlit as st
import pandas as pd
import json
import numpy as np
from pathlib import Path
from qiskit.qpy import load as qpy_load
import matplotlib.pyplot as plt

def load_ansatz(path):
    with open(path, 'rb') as f:
        circuits = qpy_load(f)
    return circuits[0] if circuits else None

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
    A_tilde = np.asarray(A_tilde)
    A = A_tilde[:N, N:N+K]
    return A

import pandas as pd

def load_pi_vector(pi_vector_path):
    """
    Load the Pi vector from a CSV file and ensure it is correctly shaped.
    """
    # Read the CSV as a single column vector
    pi_vector_df = pd.read_csv(pi_vector_path, header=None)
    
    # Ensure the vector is in the correct shape
    pi_vector = pi_vector_df.values.flatten()  # Convert to a flat array
    
    print(f"Loaded Pi vector of shape: {pi_vector.shape}")
    
    return pi_vector


def calculate_qp_distance(S_matrix, qp_vector, Pi_vector, N, K):
    """
    Calculate the norm || S @ qp_vector - Pi_vector ||.
    """
    # Extract the original S matrix
    S = extract_original_matrix(S_matrix, N, K)
    Pi = Pi_vector[:N]

    # Calculate the difference vector and its norm
    S_qp_vector = S @ qp_vector

    diff_vector = S_qp_vector- Pi
    distance_value = np.linalg.norm(diff_vector)

    dot_product = np.dot(S_qp_vector, Pi)
    # Calculate the norms of S_qp_vector and Pi
    norm_S_qp_vector = np.linalg.norm(S_qp_vector)
    norm_Pi = np.linalg.norm(Pi)
    
    # Calculate the overlap
    overlap_value = abs(dot_product) / (norm_S_qp_vector * norm_Pi)

    return distance_value, overlap_value

def plot_vqe_results(vqe_output, qp_distance=None, qp_overlap = None):
    """
    Plots the VQE results with precise titles reflecting theoretical and sampled approaches.
    If qp_distance is provided, it adds a red dashed line to the corresponding distance plot.
    """
    plots = {
        'vqe_cost_function': [],
        'vqe_sampling_cost_function': [],
    }

    custom_titles = {
        'vqe_cost_function_cost_history': 'Theoretical VQE (C_th): Cost Function Evolution',
        'vqe_sampling_cost_function_cost_history': 'Sampled VQE (C_s): Cost Function Evolution',
        'vqe_cost_function_overlap_history': 'Theoretical VQE: 1 - Overlap Evolution',
        'vqe_sampling_cost_function_overlap_history': 'Sampled VQE: 1 - Overlap Evolution',
        'vqe_cost_function_proba_lost_history': 'Theoretical VQE: Lost Probability Evolution',
        'vqe_sampling_cost_function_proba_lost_history': 'Sampled VQE: Lost Probability Evolution',
        'constrained_distance_with_Pi_for_vqe_cost_function_history': 'Theoretical VQE: Distance from Target State Evolution',
        'constrained_distance_with_Pi_for_vqe_sampling_cost_function_history': 'Sampled VQE: Distance from Target State Evolution'
    }

    for key, values in vqe_output.items():
        title = custom_titles.get(key, key.replace('_', ' ').title())
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if len(values) > 1:
            ax.plot(values, marker='o')
        else:
            ax.scatter(0, values[0], marker='o')
        
        ax.set_yscale("log")
        ax.set_title(title, fontsize=18)
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel("Value", fontsize=16)
        
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add the qp_distance as a red dashed line if applicable
        if qp_distance is not None and 'distance' in key:
            ax.axhline(qp_distance, color='red', linestyle='--', label='QP Solution Distance')
            ax.legend()

        # if qp_overlap is not None and 'overlap' in key:
        #     ax.axhline(qp_overlap, color='red', linestyle='--', label='QP Solution Overlap')
        #     ax.legend()

        plt.tight_layout()

        if 'sampling' in key:
            plots['vqe_sampling_cost_function'].append((fig, title))
        else:
            plots['vqe_cost_function'].append((fig, title))

    return plots


def main(selected_run_dir):
    # Center-aligned header
    st.markdown("<h1 style='text-align: center;'>VQE Optimization Results</h1>", unsafe_allow_html=True)

    selected_run_path = Path(selected_run_dir)
    vqe_output_path = selected_run_path / "vqe_output.json"
    ansatz_path = selected_run_path / "ansatz.qpy"
    
    # Load VQE output
    with open(vqe_output_path, 'r') as json_file:
        vqe_output = json.load(json_file)
    
    # Load QP solution vector
    qp_solution_path = selected_run_path / "qp_solution_vector.csv"
    qp_vector = pd.read_csv(qp_solution_path, header=None).values.flatten()
    
    # Load necessary matrices and vectors
    S_matrix = pd.read_csv(selected_run_path / "S_matrix.csv").values
    pi_vector_path = selected_run_path / "Pi_vector.csv"
    Pi_vector = load_pi_vector(pi_vector_path)


    # Load model configuration to get N and K
    model_config_path = selected_run_path / "model_config.json"
    with open(model_config_path, 'r') as json_file:
        model_config = json.load(json_file)
    N = model_config["N"]
    K = model_config["K"]

    # Calculate the norm || S @ qp_vector - Pi_vector ||
    qp_distance, qp_overlap= calculate_qp_distance(S_matrix, qp_vector, Pi_vector, N, K)

    # Load and display the ansatz
    ansatz = load_ansatz(ansatz_path)
    if ansatz:
        st.markdown("<h2 style='text-align: center;'>Decomposed Ansatz Circuit</h2>", unsafe_allow_html=True)
        st.pyplot(ansatz.decompose().draw(output="mpl"))

    # Plot the results from VQE output
    plots = plot_vqe_results(vqe_output, qp_distance=qp_distance, qp_overlap=1-qp_overlap)
    st.write("The QP solution distance is:", qp_distance)
    st.write("The QP solution overlap is:", 1-qp_overlap)

    # Display the plots
    st.markdown("<h2 style='text-align: center;'>VQE Optimization Results</h2>", unsafe_allow_html=True)
    plot_columns = st.columns(2)

    for col, func_type in zip(plot_columns, ['vqe_cost_function', 'vqe_sampling_cost_function']):
        for fig, title in plots[func_type]:
            with col:
                st.subheader(title)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
