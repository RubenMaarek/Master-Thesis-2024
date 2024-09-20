import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import typer
from qiskit.quantum_info import Statevector
from collections import OrderedDict
from qiskit.qpy import dump as qpy_dump
import contextlib

# Add the root directory (VARIATIONAL-MARTINGALE-SOLVER) to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from lib.config import lp_boundary_funcs, performance_funcs
from lib.vqe import create_variational_circuit, run_vqe
from lib.alg import get_probas

def run_vqe_for_lp_funcs_part1(model_config_path: Path = typer.Option(...), output_path: Path = typer.Option(...),
                               depth: int = typer.Option(1), timestamp: str = typer.Option(None)):

    os.makedirs(output_path.parent, exist_ok=True)

    with open(model_config_path, 'r') as json_file:
        model_config = json.load(json_file)
    
    S = pd.read_csv(output_path.parent / "S_matrix.csv").values
    Pi = pd.read_csv(output_path.parent / "Pi_vector.csv", header=None).values.flatten()
    D = pd.read_csv(output_path.parent / "D_matrix.csv").values
    
    n_qubits = model_config['n_qubits']
    i = model_config['i']
    initial_theta = np.random.uniform(0, 2 * np.pi, n_qubits * (depth + 1))
    
    # Generate the variational circuit (ansatz)
    ansatz = create_variational_circuit(n_qubits, initial_theta, depth)

    # Save the ansatz to a QPY file
    ansatz_path = output_path.parent / "ansatz.qpy"
    with open(ansatz_path, 'wb') as f:
        qpy_dump(ansatz, f)

    # Run VQE with the generated ansatz using the lp_boundary_funcs
    results = run_vqe(
        n_qubits, S, Pi, D, i, initial_theta, depth, performance_funcs=performance_funcs, cost_funcs=lp_boundary_funcs
    )

    history_dict_lp = results["history_dict"]

    # Define the desired order of keys for lp_boundary_funcs
    ordered_keys = [
        "vqe_cost_function_for_dmin_cost_history",
        "vqe_cost_function_for_dmax_cost_history",
        "vqe_sampling_cost_function_for_dmin_cost_history",
        "vqe_sampling_cost_function_for_dmax_cost_history",
        "vqe_cost_function_for_dmin_overlap_history",
        "vqe_cost_function_for_dmax_overlap_history",
        "vqe_sampling_cost_function_for_dmin_overlap_history",
        "vqe_sampling_cost_function_for_dmax_overlap_history",
        "vqe_cost_function_for_dmin_proba_lost_history",
        "vqe_cost_function_for_dmax_proba_lost_history",
        "vqe_sampling_cost_function_for_dmin_proba_lost_history",
        "vqe_sampling_cost_function_for_dmax_proba_lost_history",
    ]

    # Reorder the history_dict_lp
    ordered_history_dict_lp = OrderedDict((key, history_dict_lp[key]) for key in ordered_keys if key in history_dict_lp)

    # Save the ordered history dictionary to JSON
    with open(output_path, 'w') as json_file:
        json.dump(ordered_history_dict_lp, json_file, indent=4)

    # Save the final statevectors to a CSV file
    statevectors_lp_df = pd.DataFrame({key: vec.data.tolist() for key, vec in results["final_statevectors"].items()})
    statevectors_lp_df.to_csv(output_path.parent / "final_statevectors_lp.csv", index=False)

    # Save the final thetas to a CSV file
    final_thetas_lp_df = pd.DataFrame({key: theta for key, theta in results["final_thetas"].items()})
    final_thetas_lp_df.to_csv(output_path.parent / "final_thetas_lp.csv", index=False)

    # Save optimization times and shots used to JSON
    with open(output_path.parent / "optimization_times_lp.json", 'w') as json_file:
        json.dump(results["optimization_times"], json_file, indent=4)
    
    with open(output_path.parent / "optimization_methods_lp.json", 'w') as json_file:
        json.dump(results["optimization_methods"], json_file, indent=4)

    with open(output_path.parent / "shots_used_lp.json", 'w') as json_file:
        json.dump(results["shots_used_dict"], json_file, indent=4)


def run_vqe_for_lp_funcs_part2(model_config_path: Path = typer.Option(...), output_path: Path = typer.Option(...)):
    """
    This script calculates the option prices using the LP methods and stores them in a CSV file.
    Args:
        model_config_path (Path): The path to the model configuration JSON file.
        output_path (Path): The output path where the results should be saved.
    """
    # Load the model configuration
    with open(model_config_path, 'r') as json_file:
        model_config = json.load(json_file)
    
    # Load matrices and vectors needed for calculations
    S = pd.read_csv(output_path.parent / "S_matrix.csv").values
    D = pd.read_csv(output_path.parent / "D_matrix.csv").values
    i = model_config["i"]

    # Load the final statevectors generated in Part 1
    final_statevectors_lp = pd.read_csv(output_path.parent / "final_statevectors_lp.csv").to_dict(orient="list")

    # Calculate probabilities for different functions
    probas_vector_lp_min = get_probas(Statevector(np.array(final_statevectors_lp["vqe_cost_function_for_dmin"])))
    probas_vector_lp_min_sampling = get_probas(Statevector(np.array(final_statevectors_lp["vqe_sampling_cost_function_for_dmin"])))

    probas_vector_lp_max = get_probas(Statevector(np.array(final_statevectors_lp["vqe_cost_function_for_dmax"])))
    probas_vector_lp_max_sampling = get_probas(Statevector(np.array(final_statevectors_lp["vqe_sampling_cost_function_for_dmax"])))

    # Minimum boundary estimated from VQE_lp
    d_min_normal = D[i] @ probas_vector_lp_min
    d_min_sampling = D[i] @ probas_vector_lp_min_sampling
   
    # Maximum boundary estimated from VQE_lp
    d_max_normal = D[i] @ probas_vector_lp_max
    d_max_sampling = D[i] @ probas_vector_lp_max_sampling

    # Prepare a DataFrame to store all LP-based option prices
    lp_option_prices_df = pd.DataFrame({
        'Index': np.arange(len(D)),
        'd_min_normal': d_min_normal.flatten(),
        'd_max_normal': d_max_normal.flatten(),
        'd_min_sampling': d_min_sampling.flatten(),
        'd_max_sampling': d_max_sampling.flatten(),
    })

    # Save the LP option prices to a CSV file
    lp_option_prices_df.to_csv(output_path.parent / "lp_option_prices.csv", index=False)

    print("LP-based option prices have been saved to 'lp_option_prices.csv'.")

if __name__ == "__main__":
    typer.run(run_vqe_for_lp_funcs_part1)
    typer.run(run_vqe_for_lp_funcs_part2)