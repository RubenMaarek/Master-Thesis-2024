import streamlit as st
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.ticker as ticker
from qiskit.quantum_info import Statevector

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def get_probas_from_csv(csv_series):
    complex_list = [complex(num) for num in csv_series]
    psi = Statevector(complex_list)
    proba_vector = np.abs(psi.data) ** 2
    return proba_vector

def process_data(model_base_path):
    results = {
        'd_min': {}, 'vqe_price': {}, 'd_max': {}
    }

    run_dirs = [d for d in model_base_path.iterdir() if d.is_dir()]
    
    for dir_path in run_dirs:
        match = re.search(r'run_qubits_(\d+)_depth_(\d+)_iteration_(\d+)', str(dir_path.name))
        if match:
            n_qubits, depth, _ = map(int, match.groups())

            # Load necessary files
            model_config = load_json(dir_path / "model_config.json")
            N = model_config["N"]
            i = model_config["i"]

            all_option_prices_df = pd.read_csv(dir_path / "all_option_prices.csv")
            D_matrix = pd.read_csv(dir_path / "D_matrix.csv").values
            final_statevectors_df = pd.read_csv(dir_path / "final_statevectors.csv")
            final_statevectors_lp_df = pd.read_csv(dir_path / "final_statevectors_lp.csv")

            # Calculate VQE Option Prices
            vqe_probas = get_probas_from_csv(final_statevectors_df['vqe_cost_function'])
            vqe_option_prices = (D_matrix @ vqe_probas)[:N]
            vqe_price = vqe_option_prices[i]

            # Calculate LP min/max Option Prices
            d_min = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_cost_function_for_dmin'])
            d_max = D_matrix[i] @ get_probas_from_csv(final_statevectors_lp_df['vqe_cost_function_for_dmax'])

            # Store the results
            results['d_min'].setdefault((n_qubits, depth), []).append(d_min)
            results['vqe_price'].setdefault((n_qubits, depth), []).append(vqe_price)
            results['d_max'].setdefault((n_qubits, depth), []).append(d_max)

    # Calculate average values for each qubit-depth pair
    averaged_results = {}
    for key, data in results.items():
        averaged_results[key] = {
            qubit_depth: np.mean(values) for qubit_depth, values in data.items()
        }
    return averaged_results

def plot_metric(data_min, data_vqe, data_max, title, ylabel, log_scale=False, scientific_y=True):
    if not data_min or not data_vqe or not data_max:
        st.warning(f"No data available for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    qubits = sorted(set(k[0] for k in data_min.keys()))
    depths = sorted(set(k[1] for k in data_min.keys()))

    for depth in depths:
        x = []
        y_min = []
        y_vqe = []
        y_max = []
        for qubit in qubits:
            if (qubit, depth) in data_min and (qubit, depth) in data_vqe and (qubit, depth) in data_max:
                x.append(qubit)
                y_min.append(data_min[(qubit, depth)])
                y_vqe.append(data_vqe[(qubit, depth)])
                y_max.append(data_max[(qubit, depth)])
        
        # Plot the lines for d_min, vqe_price, and d_max
        ax.plot(x, y_min, '-o', label=f'D_min Depth {depth}', color='blue')
        ax.plot(x, y_vqe, '-o', label=f'VQE Price Depth {depth}', color='green')
        ax.plot(x, y_max, '-o', label=f'D_max Depth {depth}', color='red')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel(ylabel)
    ax.set_yscale('symlog')
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
    st.title("LP Average D_min, VQE Price, and D_max by Qubits and Reps")

    model_base_path = Path("model/test_2024_09_06")
    if not model_base_path.exists():
        st.error(f"Directory not found: {model_base_path}")
        return

    results = process_data(model_base_path)

    if not any(results.values()):
        st.error("No data found. Please check the directory structure and file contents.")
        return

    # Plot metrics with uniform y-axis format
    plot_metric(results['d_min'], results['vqe_price'], results['d_max'], 
                'Average D_min, VQE Price, and D_max vs. Number of Qubits', 
                'Average Values')

if __name__ == "__main__":
    main()
