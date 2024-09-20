import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.qpy import dump as qpy_dump
import typer
from collections import OrderedDict
import contextlib

# Add the root directory (VARIATIONAL-MARTINGALE-SOLVER) to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from lib.config import cost_funcs, performance_funcs
from lib.vqe import run_vqe, create_variational_circuit

def vqe_script(model_config_path: Path = typer.Option(...), output_path: Path = typer.Option(...),
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

    # Run VQE with the generated ansatz using the cost_funcs
    results = run_vqe(
        n_qubits, S, Pi, D, i, initial_theta, depth, performance_funcs=performance_funcs, cost_funcs=cost_funcs
    )

    history_dict = results["history_dict"]

    # Define the desired order of keys
    ordered_keys = [
        "vqe_cost_function_cost_history",
        "vqe_sampling_cost_function_cost_history",
        "vqe_cost_function_overlap_history",
        "vqe_sampling_cost_function_overlap_history",
        "vqe_cost_function_proba_lost_history",
        "vqe_sampling_cost_function_proba_lost_history",
        "constrained_overlap_with_Pi_for_vqe_cost_function_history",
        "constrained_overlap_with_Pi_for_vqe_sampling_cost_function_history",
        "constrained_distance_with_Pi_for_vqe_cost_function_history",
        "constrained_distance_with_Pi_for_vqe_sampling_cost_function_history"
    ]

    # Reorder the history_dict
    ordered_history_dict = OrderedDict((key, history_dict[key]) for key in ordered_keys if key in history_dict)

    # Save the ordered history dictionary to JSON
    with open(output_path, 'w') as json_file:
        json.dump(ordered_history_dict, json_file, indent=4)

    # Save the final statevectors to a CSV file
    statevectors_df = pd.DataFrame({key: vec.data.tolist() for key, vec in results["final_statevectors"].items()})
    statevectors_df.to_csv(output_path.parent / "final_statevectors.csv", index=False)

    # Save the final thetas to a CSV file
    final_thetas_df = pd.DataFrame({key: theta for key, theta in results["final_thetas"].items()})
    final_thetas_df.to_csv(output_path.parent / "final_thetas.csv", index=False)

    # Save optimization times and shots used to JSON
    with open(output_path.parent / "optimization_times.json", 'w') as json_file:
        json.dump(results["optimization_times"], json_file, indent=4)
    
    # Save optimization methods to JSON (optional)
    with open(output_path.parent / "optimization_methods.json", 'w') as json_file:
        json.dump(results["optimization_methods"], json_file, indent=4)

    with open(output_path.parent / "shots_used.json", 'w') as json_file:
        json.dump(results["shots_used_dict"], json_file, indent=4)

if __name__ == "__main__":
    typer.run(vqe_script)
