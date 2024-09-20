import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
from sklearn.metrics import mean_squared_error
from qiskit.quantum_info import Statevector

# Helper functions
def extract_numbers_from_dirname(dirname):
    match = re.search(r'run_qubits_(\d+)_depth_(\d+)_iteration_(\d+)', str(dirname))
    return int(match.group(1)) if match else None

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def get_probas_from_csv(csv_series):
    complex_list = [complex(num) for num in csv_series]
    psi = Statevector(complex_list)
    proba_vector = np.abs(psi.data) ** 2
    return proba_vector

# Data processing functions
def process_directories():
    model_base_path = Path("model/test_2024_09_06")
    run_dirs = [d for d in model_base_path.iterdir() if d.is_dir()]
    
    if not run_dirs:
        st.error("No directories found in the model folder.")
        return None
    
    run_dirs_with_numbers = [(d, extract_numbers_from_dirname(d)) for d in run_dirs if extract_numbers_from_dirname(d) is not None]
    
    if not run_dirs_with_numbers:
        st.error("No valid directories matching the pattern found.")
        return None
    
    grouped_dirs = {}
    for d, n_qubits in run_dirs_with_numbers:
        key = n_qubits
        if key not in grouped_dirs:
            grouped_dirs[key] = []
        grouped_dirs[key].append(d)
    
    return grouped_dirs

# Data processing function
def process_data(grouped_dirs):
    results = {
        'optimization_times_vqe': {}, 'optimization_times_vqe_sampling': {},
        'cpu_times_vqe': {}, 'cpu_times_vqe_sampling': {},
        'shots_used_vqe': {}, 'shots_used_vqe_sampling': {},
        'rmse_vqe_mc': {}, 'rmse_vqe_sampling_mc': {}, 'rmse_qp_mc': {}
    }

    for n_qubits, dirs in grouped_dirs.items():
        all_vqe_real_times, all_vqe_sampling_real_times = [], []
        all_vqe_cpu_times, all_vqe_sampling_cpu_times = [], []
        all_vqe_shots, all_vqe_sampling_shots = [], []
        all_rmse_vqe_mc, all_rmse_vqe_sampling_mc = [], []
        all_rmse_qp_mc = []

        for d in dirs:
            selected_run_path = Path(d)
            
            # Load necessary files
            model_config = load_json(selected_run_path / "model_config.json")
            all_option_prices_df = pd.read_csv(selected_run_path / "all_option_prices.csv")
            D_matrix = pd.read_csv(selected_run_path / "D_matrix.csv").values
            final_statevectors_df = pd.read_csv(selected_run_path / "final_statevectors.csv")
            optimization_metrics = load_json(selected_run_path / "optimization_times.json")
            shots_metrics = load_json(selected_run_path / "shots_used.json")

            N = model_config["N"]

            # Calculate VQE Option Prices
            vqe_probas = get_probas_from_csv(final_statevectors_df['vqe_cost_function'])
            vqe_sampling_probas = get_probas_from_csv(final_statevectors_df['vqe_sampling_cost_function'])

            vqe_option_prices = D_matrix @ vqe_probas
            vqe_sampling_option_prices = D_matrix @ vqe_sampling_probas

            vqe_option_prices = vqe_option_prices[:N]
            vqe_sampling_option_prices = vqe_sampling_option_prices[:N]    

            # Calculate RMSE
            rmse_vqe_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], vqe_option_prices))
            rmse_vqe_sampling_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], vqe_sampling_option_prices))
            rmse_qp_mc = np.sqrt(mean_squared_error(all_option_prices_df['MC_Option_Prices'], all_option_prices_df['QP_Option_Prices']))

            # Normalize RMSE
            norm_rmse_vqe_mc = rmse_vqe_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100
            norm_rmse_vqe_sampling_mc = rmse_vqe_sampling_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100
            norm_rmse_qp_mc = rmse_qp_mc / np.mean(all_option_prices_df['MC_Option_Prices']) * 100

            # Append results
            all_rmse_vqe_mc.append(norm_rmse_vqe_mc)
            all_rmse_vqe_sampling_mc.append(norm_rmse_vqe_sampling_mc)
            all_rmse_qp_mc.append(norm_rmse_qp_mc)

            if optimization_metrics:
                vqe_metrics = optimization_metrics.get('vqe_cost_function', {})
                vqe_sampling_metrics = optimization_metrics.get('vqe_sampling_cost_function', {})
                
                all_vqe_real_times.append(vqe_metrics.get('real_world_time', np.nan))
                all_vqe_sampling_real_times.append(vqe_sampling_metrics.get('real_world_time', np.nan))
                
                all_vqe_cpu_times.append(vqe_metrics.get('cpu_time', np.nan))
                all_vqe_sampling_cpu_times.append(vqe_sampling_metrics.get('cpu_time', np.nan))

            if shots_metrics:
                all_vqe_shots.append(shots_metrics.get('vqe_cost_function', np.nan))
                all_vqe_sampling_shots.append(shots_metrics.get('vqe_sampling_cost_function', np.nan))

        # Aggregate results
        for metric_name, metric_data in [
            ('optimization_times_vqe', all_vqe_real_times),
            ('optimization_times_vqe_sampling', all_vqe_sampling_real_times),
            ('cpu_times_vqe', all_vqe_cpu_times),
            ('cpu_times_vqe_sampling', all_vqe_sampling_cpu_times),
            ('shots_used_vqe', all_vqe_shots),
            ('shots_used_vqe_sampling', all_vqe_sampling_shots),
            ('rmse_vqe_mc', all_rmse_vqe_mc),
            ('rmse_vqe_sampling_mc', all_rmse_vqe_sampling_mc)
        ]:
            if metric_data:
                results[metric_name][n_qubits] = (np.mean(metric_data), np.std(metric_data))

        if all_rmse_qp_mc:
            results['rmse_qp_mc'][n_qubits] = (np.mean(all_rmse_qp_mc), np.std(all_rmse_qp_mc))

    return results

# Plotting function (unchanged)
def plot_metric(qubits, data_vqe, data_vqe_sampling, title, ylabel, log_scale=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(qubits, [d[0] for d in data_vqe], yerr=[d[1] for d in data_vqe], 
                marker='o', label='VQE', capsize=5, markersize=8, linewidth=2)
    ax.errorbar(qubits, [d[0] for d in data_vqe_sampling], yerr=[d[1] for d in data_vqe_sampling], 
                marker='x', linestyle='--', label='VQE Sampling', capsize=5, markersize=8, linewidth=2)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    st.pyplot(fig)

def plot_qp_mc_metric(qubits, data_qp_mc, title, ylabel, log_scale=False):
    """
    Plots the RMSE for QP vs MC over the number of qubits.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot QP vs MC RMSE
    ax.errorbar(qubits, [d[0] for d in data_qp_mc], yerr=[d[1] for d in data_qp_mc],
                marker='o', label='QP vs MC', capsize=5, markersize=8, linewidth=2, color='blue')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.legend()
    st.pyplot(fig)

# Plotting function for RMSE VQE, VQE Sampling, and QP vs MC
def plot_rmse_comparison(qubits, data_vqe, data_vqe_sampling, data_qp, title, ylabel, log_scale=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot VQE RMSE
    ax.errorbar(qubits, [d[0] for d in data_vqe], yerr=[d[1] for d in data_vqe],
                marker='o', label='VQE vs MC', capsize=5, markersize=8, linewidth=2, color='blue')

    # Plot VQE Sampling RMSE
    ax.errorbar(qubits, [d[0] for d in data_vqe_sampling], yerr=[d[1] for d in data_vqe_sampling],
                marker='x', linestyle='--', label='VQE Sampling vs MC', capsize=5, markersize=8, linewidth=2, color='orange')

    # Plot QP RMSE
    ax.errorbar(qubits, [d[0] for d in data_qp], yerr=[d[1] for d in data_qp],
                marker='s', linestyle=':', label='QP vs MC', capsize=3, markersize=5, linewidth=1, color='red')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if log_scale:
        ax.set_yscale("log")
    
    ax.legend()
    st.pyplot(fig)

# Main function
def main():
    st.title("Quantum Algorithm Performance Metrics")

    grouped_dirs = process_directories()
    if not grouped_dirs:
        return

    results = process_data(grouped_dirs)
    qubits = sorted(results['optimization_times_vqe'].keys())

    # Plot Average Real-World Time vs. Number of Qubits
    plot_metric(qubits, [results['optimization_times_vqe'][q] for q in qubits],
                [results['optimization_times_vqe_sampling'][q] for q in qubits],
                'Time-to-Solution vs. Number of Qubits', 'Average Time (s)')

    # Plot Average CPU Time vs. Number of Qubits
    plot_metric(qubits, [results['cpu_times_vqe'][q] for q in qubits],
                [results['cpu_times_vqe_sampling'][q] for q in qubits],
                'CPU Time vs. Number of Qubits', 'Average CPU Time (s)')

    # Plot Average Shots Used vs. Number of Qubits
    plot_metric(qubits, [results['shots_used_vqe'][q] for q in qubits],
                [results['shots_used_vqe_sampling'][q] for q in qubits],
                'Shots Used vs. Number of Qubits', 'Average Shots Used')

    # New plot: RMSE comparison for VQE, VQE Sampling, and QP vs MC
    plot_rmse_comparison(qubits,
                         [results['rmse_vqe_mc'][q] for q in qubits],
                         [results['rmse_vqe_sampling_mc'][q] for q in qubits],
                         [results['rmse_qp_mc'][q] for q in qubits],
                         'Normalized RMSE vs. Number of Qubits (VQE, VQE Sampling, and QP vs MC)', 'Normalized RMSE (%)')


if __name__ == "__main__":
    main()
