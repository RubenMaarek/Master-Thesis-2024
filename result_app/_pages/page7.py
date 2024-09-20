import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.ticker as ticker

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def process_data(model_base_path):
    results = {
        'dmin_loss': {}, 'dmax_loss': {}, 'dmin_overlap': {}, 'dmax_overlap': {}, 'dmin_proba_loss': {}, 'dmax_proba_loss': {}
    }

    run_dirs = [d for d in model_base_path.iterdir() if d.is_dir()]
    
    for dir_path in run_dirs:
        match = re.search(r'run_qubits_(\d+)_depth_(\d+)_iteration_(\d+)', str(dir_path.name))
        if match:
            n_qubits, depth, _ = map(int, match.groups())
            vqe_output = load_json(dir_path / "vqe_lp_output.json")  # Load the LP output file
            if vqe_output:
                try:
                    # Append the values to lists keyed by (n_qubits, depth)
                    for key, metric in [
                        ('dmin_loss', 'vqe_cost_function_for_dmin_cost_history'),
                        ('dmax_loss', 'vqe_cost_function_for_dmax_cost_history'),
                        ('dmin_overlap', 'vqe_cost_function_for_dmin_overlap_history'),
                        ('dmax_overlap', 'vqe_cost_function_for_dmax_overlap_history'),
                        ('dmin_proba_loss', 'vqe_cost_function_for_dmin_proba_lost_history'),
                        ('dmax_proba_loss', 'vqe_cost_function_for_dmax_proba_lost_history')
                    ]:
                        results[key].setdefault((n_qubits, depth), []).append(vqe_output[metric][-1])
                    
                except (KeyError, IndexError) as e:
                    st.warning(f"Error processing data from {dir_path}: {str(e)}")
                    continue

    # Average and standard deviation of the results over all iterations for each qubit-depth combination
    averaged_results = {}
    for key, data in results.items():
        averaged_results[key] = {
            qubit_depth: (np.mean(values), np.std(values)) for qubit_depth, values in data.items()
        }
    return averaged_results


def plot_metric(data, title, ylabel, log_scale=False, scientific_y=True):
    if not data:
        st.warning(f"No data available for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qubits = sorted(set(k[0] for k in data.keys()))
    depths = sorted(set(k[1] for k in data.keys()))
    
    for depth in depths:
        x = []
        y_mean = []
        y_std = []
        for qubit in qubits:
            if (qubit, depth) in data:
                x.append(qubit)
                # Calculate the mean and standard deviation for the current qubit-depth pair
                mean_val, std_val = data[(qubit, depth)]
                y_mean.append(mean_val)
                y_std.append(std_val)
        
        # Plot the mean line with error bars representing standard deviation
        ax.errorbar(x, y_mean, yerr=y_std, fmt='-o', capsize=5, markersize=5, label=f'Depth {depth}')
    
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
    st.title("LP Average Results by Qubits and Reps")

    model_base_path = Path("model/test_2024_09_06")
    if not model_base_path.exists():
        st.error(f"Directory not found: {model_base_path}")
        return

    results = process_data(model_base_path)

    if not any(results.values()):
        st.error("No data found. Please check the directory structure and file contents.")
        return

    # Plot metrics with uniform y-axis format
    plot_metric(results['dmin_loss'], 'Average Final Loss Value (dmin) vs. Number of Qubits', 'Average Final Loss Value')
    plot_metric(results['dmax_loss'], 'Average Final Loss Value (dmax) vs. Number of Qubits', 'Average Final Loss Value')
    plot_metric(results['dmin_overlap'], 'Average Overlap (dmin) vs. Number of Qubits', 'Average Overlap')
    plot_metric(results['dmax_overlap'], 'Average Overlap (dmax) vs. Number of Qubits', 'Average Overlap')
    plot_metric(results['dmin_proba_loss'], 'Average Probability of Loss (dmin) vs. Number of Qubits', 'Average Probability of Loss')
    plot_metric(results['dmax_proba_loss'], 'Average Probability of Loss (dmax) vs. Number of Qubits', 'Average Probability of Loss')

if __name__ == "__main__":
    main()

