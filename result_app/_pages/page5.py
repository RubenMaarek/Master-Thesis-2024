import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.ticker as ticker
import matplotlib.cm as cm

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def extract_original_matrix(A_tilde, N, K):
    A_tilde = np.asarray(A_tilde)
    A = A_tilde[:N, N:N+K]
    return A

def calculate_qp_metrics(S_matrix, qp_vector, Pi_vector, N, K):
    S = extract_original_matrix(S_matrix, N, K)
    Pi = Pi_vector[:N]

    S_qp_vector = S @ qp_vector
    diff_vector = S_qp_vector - Pi
    distance_value = np.linalg.norm(diff_vector)

    dot_product = np.dot(S_qp_vector, Pi)
    norm_S_qp_vector = np.linalg.norm(S_qp_vector)
    norm_Pi = np.linalg.norm(Pi)
    
    overlap_value = 1 - abs(dot_product) / (norm_S_qp_vector * norm_Pi)

    qp_proba_loss = 1 - np.sum(qp_vector)

    return distance_value, overlap_value, qp_proba_loss

def process_data(model_base_path):
    results = {
        'loss': {}, 'overlap': {}, 'proba_loss': {},
        'constrained_overlap': {}, 'constrained_distance': {}
    }

    sampling_results = {
        'loss': {}, 'overlap': {}, 'proba_loss': {},
        'constrained_overlap': {}, 'constrained_distance': {}
    }
    
    qp_distances = {}
    qp_overlaps = {}
    qp_proba_losses = {}

    run_dirs = [d for d in model_base_path.iterdir() if d.is_dir()]
    
    for dir_path in run_dirs:
        match = re.search(r'run_qubits_(\d+)_depth_(\d+)_iteration_(\d+)', str(dir_path.name))
        if match:
            n_qubits, depth, _ = map(int, match.groups())
            vqe_output = load_json(dir_path / "vqe_output.json")
            if vqe_output:
                try:
                    # Append the values to lists keyed by (n_qubits, depth) for the exact VQE cost function
                    for key, metric in [
                        ('loss', 'vqe_cost_function_cost_history'),
                        ('overlap', 'vqe_cost_function_overlap_history'),
                        ('proba_loss', 'vqe_cost_function_proba_lost_history')
                    ]:
                        results[key].setdefault((n_qubits, depth), []).append(vqe_output[metric][-1])
                    
                    # Append the values to lists keyed by (n_qubits, depth) for the sampling cost function
                    for key, metric in [
                        ('loss', 'vqe_sampling_cost_function_cost_history'),
                        ('overlap', 'vqe_sampling_cost_function_overlap_history'),
                        ('proba_loss', 'vqe_sampling_cost_function_proba_lost_history')
                    ]:
                        sampling_results[key].setdefault((n_qubits, depth), []).append(vqe_output[metric][-1])
                    
                    # Append QP distances, overlaps, and proba losses
                    S_matrix = pd.read_csv(dir_path / "S_matrix.csv").values
                    qp_vector = pd.read_csv(dir_path / "qp_solution_vector.csv", header=None).values.flatten()
                    Pi_vector = pd.read_csv(dir_path / "Pi_vector.csv", header=None).values.flatten()

                    model_config = load_json(dir_path / "model_config.json")
                    N = model_config["N"]
                    K = model_config["K"]

                    qp_distance, qp_overlap, qp_proba_loss = calculate_qp_metrics(S_matrix, qp_vector, Pi_vector, N, K)

                    qp_distances.setdefault((n_qubits, depth), []).append(qp_distance)
                    qp_overlaps.setdefault((n_qubits, depth), []).append(qp_overlap)
                    qp_proba_losses.setdefault((n_qubits, depth), []).append(qp_proba_loss)
                    
                    if 'constrained_overlap_with_Pi_for_vqe_cost_function_history' in vqe_output:
                        results['constrained_overlap'].setdefault((n_qubits, depth), []).append(
                            vqe_output['constrained_overlap_with_Pi_for_vqe_cost_function_history'][-1])
                        sampling_results['constrained_overlap'].setdefault((n_qubits, depth), []).append(
                            vqe_output['constrained_overlap_with_Pi_for_vqe_sampling_cost_function_history'][-1])
                    
                    if 'constrained_distance_with_Pi_for_vqe_cost_function_history' in vqe_output:
                        results['constrained_distance'].setdefault((n_qubits, depth), []).append(
                            vqe_output['constrained_distance_with_Pi_for_vqe_cost_function_history'][-1])
                        sampling_results['constrained_distance'].setdefault((n_qubits, depth), []).append(
                            vqe_output['constrained_distance_with_Pi_for_vqe_sampling_cost_function_history'][-1])
                except (KeyError, IndexError) as e:
                    st.warning(f"Error processing data from {dir_path}: {str(e)}")
                    continue

    # Average and standard deviation of the results over all iterations for each qubit-depth combination
    averaged_results = {}
    averaged_sampling_results = {}
    qp_averages = {}
    
    for key, data in results.items():
        averaged_results[key] = {
            qubit_depth: (np.mean(values), np.std(values)) for qubit_depth, values in data.items()
        }
    for key, data in sampling_results.items():
        averaged_sampling_results[key] = {
            qubit_depth: (np.mean(values), np.std(values)) for qubit_depth, values in data.items()
        }

    # Compute averages for QP distances, overlaps, and proba losses
    qp_averages['constrained_distance'] = {
        qubit_depth: np.mean(values) for qubit_depth, values in qp_distances.items()
    }
    qp_averages['constrained_overlap'] = {
        qubit_depth: np.mean(values) for qubit_depth, values in qp_overlaps.items()
    }
    qp_averages['proba_loss'] = {
        qubit_depth: np.mean(values) for qubit_depth, values in qp_proba_losses.items()
    }

    return averaged_results, averaged_sampling_results, qp_averages

def plot_metric(data, sampling_data, qp_data=None, title='', ylabel='', log_scale=False, scientific_y=True):
    if not data or not sampling_data:
        st.warning(f"No data available for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qubits = sorted(set(k[0] for k in data.keys()))
    depths = sorted(set(k[1] for k in data.keys()))
    
    # Use color maps for exact and sampling depths
    exact_cmap = plt.get_cmap('Blues')
    sampling_cmap = plt.get_cmap('Oranges')
    
    num_depths = len(depths)
    
    # Loop through each depth
    for idx, depth in enumerate(depths):
        x = []
        y_mean = []
        y_std = []
        sampling_y_mean = []
        sampling_y_std = []
        qp_y_mean = []
        
        for qubit in qubits:
            if (qubit, depth) in data:
                x.append(qubit)
                # exact cost function mean and std
                mean_val = data[(qubit, depth)][0]
                std_val = data[(qubit, depth)][1]
                y_mean.append(mean_val)
                y_std.append(std_val)

                # Sampling cost function mean and std
                sampling_mean_val = sampling_data[(qubit, depth)][0]
                sampling_std_val = sampling_data[(qubit, depth)][1]
                sampling_y_mean.append(sampling_mean_val)
                sampling_y_std.append(sampling_std_val)

                # QP average (only plot if qp_data is provided)
                if qp_data and (qubit, depth) in qp_data:
                    qp_mean_val = qp_data[(qubit, depth)]
                    qp_y_mean.append(qp_mean_val)
        
        # Determine color for current depth
        exact_color = exact_cmap(float(idx) / num_depths)
        sampling_color = sampling_cmap(float(idx) / num_depths)
        
        # Plot the exact cost function mean line with error bars
        ax.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=5, markersize=5, 
                    label=f'Exact, Depth {depth}', color=exact_color)
        
        # Plot the sampling cost function mean line with error bars
        ax.errorbar(x, sampling_y_mean, yerr=sampling_y_std, fmt='-o', capsize=5, markersize=5, 
                    label=f'Sampling, Depth {depth}', color=sampling_color)

        # Plot the QP averages as dashed lines (if provided)
        if qp_data and qp_y_mean:
            ax.plot(x, qp_y_mean, '--', label=f'QP Average Depth {depth}', linewidth=2, color='purple')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Apply scientific notation uniformly across all plots
    if scientific_y:
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    if log_scale:
        ax.set_yscale("log")
        if scientific_y:
            # For log scale, we need to set the minor formatter as well
            ax.yaxis.set_minor_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))
            for label in ax.yaxis.get_minorticklabels():
                label.set_visible(False)
    
    ax.legend()
    st.pyplot(fig)


def main():
    st.title("Average Results by Qubits and Reps")

    model_base_path = Path("model/test_2024_09_06")
    if not model_base_path.exists():
        st.error(f"Directory not found: {model_base_path}")
        return

    results, sampling_results, qp_averages = process_data(model_base_path)

    if not any(results.values()) or not any(sampling_results.values()):
        st.error("No data found. Please check the directory structure and file contents.")
        return

    # Plot metrics with uniform y-axis format, comparing exact vs sampling cost functions
    plot_metric(results['loss'], sampling_results['loss'], {}, 'Average Final Loss Value vs. Number of Qubits', 'Average Final Loss Value')
    plot_metric(results['overlap'], sampling_results['overlap'], None, 'Average Overlap vs. Number of Qubits', 'Average Overlap')
    plot_metric(results['proba_loss'], sampling_results['proba_loss'],None, 'Average Lost Probability vs. Number of Qubits', 'Average Lost Probability')
    plot_metric(results['constrained_distance'], sampling_results['constrained_distance'], qp_averages['constrained_distance'], 'Average Distance vs. Number of Qubits', 'Average Distance')

if __name__ == "__main__":
    main()
