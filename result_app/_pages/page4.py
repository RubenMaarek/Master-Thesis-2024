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
    A = A_tilde[:N, N:N + K]
    return A


def plot_lp_results(vqe_lp_output, bound, qp_distance=None, qp_overlap=None):
    """
    Plots the LP results with precise titles reflecting min and max approaches.
    If qp_distance is provided, it adds a red dashed line to the corresponding distance plot.
    Filters to only plot relevant cost functions (min or max based on the bound argument).
    """
    plots = {
        'lp_min_cost_function': [],
        'lp_sampling_min_cost_function': [],
    }

    custom_titles = {
        'vqe_cost_function_for_dmin_cost_history': 'Theoretical LP (D_min): Cost Function Evolution',
        'vqe_sampling_cost_function_for_dmin_cost_history': 'Sampled LP (D_min_s): Cost Function Evolution',
        'vqe_cost_function_for_dmin_overlap_history': 'Theoretical LP (D_min): Overlap Evolution',
        'vqe_sampling_cost_function_for_dmin_overlap_history': 'Sampled LP (D_min_s): Overlap Evolution',
        'vqe_cost_function_for_dmin_proba_lost_history': 'Theoretical LP (D_min): Lost Probability Evolution',
        'vqe_sampling_cost_function_for_dmin_proba_lost_history': 'Sampled LP (D_min_s): Lost Probability Evolution',
        'constrained_distance_with_Pi_for_vqe_cost_function_for_dmin_history': 'Theoretical LP: Distance from Target State Evolution',
        'constrained_distance_with_Pi_for_vqe_sampling_cost_function_for_dmin_history': 'Sampled LP: Distance from Target State Evolution',
        'vqe_cost_function_for_dmax_cost_history': 'Theoretical LP (D_max): Cost Function Evolution',
        'vqe_sampling_cost_function_for_dmax_cost_history': 'Sampled LP (D_max_s): Cost Function Evolution',
        'vqe_cost_function_for_dmax_overlap_history': 'Theoretical LP (D_max): Overlap Evolution',
        'vqe_sampling_cost_function_for_dmax_overlap_history': 'Sampled LP (D_max_s): Overlap Evolution',
        'vqe_cost_function_for_dmax_proba_lost_history': 'Theoretical LP (D_max): Lost Probability Evolution',
        'vqe_sampling_cost_function_for_dmax_proba_lost_history': 'Sampled LP (D_max_s): Lost Probability Evolution',
        'constrained_distance_with_Pi_for_vqe_cost_function_for_dmax_history': 'Theoretical LP: Distance from Target State Evolution',
        'constrained_distance_with_Pi_for_vqe_sampling_cost_function_for_dmax_history': 'Sampled LP: Distance from Target State Evolution'
    }

    for key, values in vqe_lp_output.items():
        # Only consider keys that match the bound ('dmin' or 'dmax')
        if bound in key:
            title = custom_titles.get(key, key.replace('_', ' ').title())

            fig, ax = plt.subplots(figsize=(10, 6))

            if len(values) > 1:
                ax.plot(values, marker='o')
            else:
                ax.scatter(0, values[0], marker='o')

            if bound == "min":
                ax.set_yscale("log")
            ax.set_title(title, fontsize=14)
            ax.set_xlabel("Iteration", fontsize=12)
            ax.set_ylabel("Value", fontsize=12)

            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.grid(True, which="both", ls="-", alpha=0.2)

            # Add the qp_distance as a red dashed line if applicable
            if qp_distance is not None and 'distance' in key:
                ax.axhline(qp_distance, color='red', linestyle='--', label='QP Solution Distance')
                ax.legend()

            if qp_overlap is not None and 'overlap' in key:
                ax.axhline(qp_overlap, color='red', linestyle='--', label='QP Solution Overlap')
                ax.legend()

            plt.tight_layout()

            if 'sampling' in key:
                plots['lp_sampling_min_cost_function'].append((fig, title))
            else:
                plots['lp_min_cost_function'].append((fig, title))

    return plots


def main(selected_run_dir):
    # Center-aligned header for the page
    st.markdown("<h1 style='text-align: center;'>LP Optimization Results</h1>", unsafe_allow_html=True)

    selected_run_path = Path(selected_run_dir)
    lp_output_path = selected_run_path / "vqe_lp_output.json"
    ansatz_path = selected_run_path / "ansatz.qpy"

    # Load LP output
    with open(lp_output_path, 'r') as json_file:
        vqe_lp_output = json.load(json_file)

    # Load and display the ansatz
    ansatz = load_ansatz(ansatz_path)
    if ansatz:
        st.markdown("<h2 style='text-align: center;'>Decomposed Ansatz Circuit</h2>", unsafe_allow_html=True)
        st.pyplot(ansatz.decompose().draw(output="mpl"))

    # Plot for D_min
    st.markdown("<h2 style='text-align: center;'>LP Optimization Results (D_min)</h2>", unsafe_allow_html=True)
    plots_min = plot_lp_results(vqe_lp_output, bound="min")
    plot_columns_min = st.columns(2)

    for col, func_type in zip(plot_columns_min, ['lp_min_cost_function', 'lp_sampling_min_cost_function']):
        for fig, title in plots_min[func_type]:
            with col:
                st.subheader(title)
                st.pyplot(fig)

    # Plot for D_max
    st.markdown("<h2 style='text-align: center;'>LP Optimization Results (D_max)</h2>", unsafe_allow_html=True)
    plots_max = plot_lp_results(vqe_lp_output, bound="max")
    plot_columns_max = st.columns(2)

    for col, func_type in zip(plot_columns_max, ['lp_min_cost_function', 'lp_sampling_min_cost_function']):
        for fig, title in plots_max[func_type]:
            with col:
                st.subheader(title)
                st.pyplot(fig)


if __name__ == "__main__":
    main()
