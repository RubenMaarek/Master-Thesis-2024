from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import TwoLocal
import numpy as np
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Union
import time  # Import the time module to track execution time
import functools
import os
from lib.utils import plot
from lib.config import *


@functools.lru_cache(maxsize=1000)

def _create_variational_circuit_cached(n_qubits, theta_params_tuple, depth, entanglement_type="full"):
    theta_params = np.array(theta_params_tuple)
    """
    entanglement: Specifies the entanglement structure. Can be a string ('full',
    'linear', 'reverse_linear', 'circular' or 'sca'), a list of integer-pairs
    specifying the indices of qubits entangled with one another, or a callable
    returning such a list provided with the index of the entanglement layer.
    Default to 'full' entanglement. Note that if entanglement_blocks = 'cx',
    then 'full' entanglement provides the same unitary as 'reverse_linear'
    but the latter option has fewer entangling gates.
    """
    rotation_blocks = ["ry"]
    entanglement_blocks = "cz"

    variational_form = TwoLocal(
        n_qubits,
        rotation_blocks=rotation_blocks,
        entanglement_blocks=entanglement_blocks,
        entanglement=entanglement_type,
        reps=depth,
        parameter_prefix="θ",
    )

    if len(theta_params) != (depth + 1) * n_qubits * len(rotation_blocks):
        raise ValueError(
            "Length of theta_params must be equal to number of circuit parameters"
        )

    ansatz = QuantumCircuit(n_qubits)
    ansatz = ansatz.compose(variational_form)
    param_dict = dict(zip(ansatz.parameters, theta_params))
    ansatz = ansatz.assign_parameters(param_dict)

    return ansatz

def create_variational_circuit(n_qubits, theta_params, depth, entanglement_type="full"):
    if isinstance(theta_params, np.ndarray):
        theta_params_tuple = tuple(theta_params.flatten())
    else:
        theta_params_tuple = tuple(theta_params)
    
    return _create_variational_circuit_cached(n_qubits, theta_params_tuple, depth, entanglement_type)

def objective_function(theta_params, n_qubits, S, Pi, D, i, depth, cost_func, entanglement):
    ansatz = create_variational_circuit(n_qubits, theta_params, entanglement_type=entanglement, depth=depth)
    overlap_term, proba_lost, cost, _ = cost_func(S, Pi, D, i, ansatz)  # Discard the shots_used value
    return np.real(cost), np.real(overlap_term), np.real(proba_lost)

def callback(xk, n_qubits, S, Pi, D, i, depth, history_dict, cost_func, performance_funcs, entanglement, shots_dict):
    # Get the objective values (cost, overlap_term, proba_lost)
    f_val, overlap_term, proba_lost = objective_function(xk, n_qubits, S, Pi, D, i, depth, cost_func, entanglement)
    
    # Get the shots_used by calling the cost function directly
    _, _, _, shots_used = cost_func(S, Pi, D, i, create_variational_circuit(n_qubits, xk, entanglement_type=entanglement, depth=depth))

    # Save the shots used for this cost function only on the first iteration
    if shots_dict.get(cost_func.__name__) is None:
        shots_dict[cost_func.__name__] = shots_used
    
    # Store the values in the history dictionary
    history_dict[f"{cost_func.__name__}_cost_history"].append(f_val)
    history_dict[f"{cost_func.__name__}_overlap_history"].append(overlap_term)
    history_dict[f"{cost_func.__name__}_proba_lost_history"].append(proba_lost)
    
    ansatz = create_variational_circuit(n_qubits, xk, entanglement_type=entanglement, depth=depth)
    
    # Update performance function values
    for performance_func in performance_funcs:
        performance = performance_func(S, Pi, ansatz)
        history_dict[f"{performance_func.__name__}_for_{cost_func.__name__}_history"].append(performance)

    # Early stopping mechanism (optional)
    if len(history_dict[f"{cost_func.__name__}_cost_history"]) > 10:
        recent_costs = history_dict[f"{cost_func.__name__}_cost_history"][-10:]
        if all(abs(cost - recent_costs[0]) < 1e-6 for cost in recent_costs):
            return True
    return False


# Define optimizer settings for both methods

lbfgsb_settings = {
    "maxiter": 500,  # Maximum number of iterations
    "disp": False,  # Display convergence messages
    "ftol": 1e-6,  # Tolerance for termination by the change of function value
    "gtol": 1e-6,  # Tolerance for termination by the norm of the gradient
    "maxcor": 10,  # Number of corrections used in the L-BFGS update (typically 3-20)
    "maxfun": 15000,  # Maximum number of function evaluations
    "maxls": 20,  # Maximum number of line search steps per iteration
}

powell_settings = {
    "maxiter": 500,  # Maximum number of iterations
    "disp": False,    # Display convergence messages
    "xtol": 1e-4,    # Tolerance for termination by the change of variables
    "ftol": 1e-4,    # Tolerance for termination by the change of function value
}

def run_vqe(
    n_qubits,
    S,
    Pi,
    D,
    i,
    initial_theta,
    depth,
    entanglement="full",
    performance_funcs=performance_funcs,
    cost_funcs=cost_funcs,
):
    """
    Runs the VQE and stores the values of the overlap term, proba_lost, and cost function.
    Also tracks the time taken for each optimization process, including both real-world (wall-clock) time and CPU time.
    Additionally, stores the optimization method used for each cost function.
    """
    if S.shape[0] != 2**n_qubits or S.shape[1] != 2**n_qubits or len(Pi) != 2**n_qubits:
        raise ValueError("System size does not match qubits")

    # Initialize history dictionaries
    history_dict = {f"{func.__name__}_cost_history": [] for func in cost_funcs}
    history_dict.update({f"{func.__name__}_overlap_history": [] for func in cost_funcs})
    history_dict.update({f"{func.__name__}_proba_lost_history": [] for func in cost_funcs})

    for cost_func in cost_funcs:
        for performance_func in performance_funcs:
            history_dict[f"{performance_func.__name__}_for_{cost_func.__name__}_history"] = []

    final_values = {}
    final_performances = {}
    final_statevectors = {}
    final_thetas = {}
    final_ansatzes = {}
    optimization_times = {}
    optimization_methods = {}  # New dictionary to store optimization methods
    shots_used_dict = {}

    for cost_func in cost_funcs:
        print(f"Starting optimization for cost function: {cost_func.__name__}")
        
        if "sampling" in cost_func.__name__:
            optimizer_method = "Powell"
            optimizer_settings = powell_settings
        else:
            optimizer_method = "L-BFGS-B"
            optimizer_settings = lbfgsb_settings

        # Store the optimization method used
        optimization_methods[cost_func.__name__] = optimizer_method

        # Start timers for real-world time and CPU time
        start_time = time.time()
        start_cpu_time = os.times().user

        # Run the optimization
        rez = minimize(
            lambda theta: objective_function(theta, n_qubits, S, Pi, D, i, depth, cost_func, entanglement)[0],
            initial_theta,
            method=optimizer_method,
            callback=lambda xk, cost_func=cost_func: callback(
                xk, n_qubits, S, Pi, D, i, depth, history_dict, cost_func, performance_funcs, entanglement, shots_used_dict
            ),
            options=optimizer_settings,
        )

        # End timers and calculate durations
        end_time = time.time()
        end_cpu_time = os.times().user

        optimization_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time

        # Store both real-world and CPU times
        optimization_times[cost_func.__name__] = {
            "real_world_time": optimization_time,
            "cpu_time": cpu_time
        }

        final_values[cost_func.__name__] = rez.fun
        final_performances[cost_func.__name__] = {
            performance_func.__name__: history_dict[f"{performance_func.__name__}_for_{cost_func.__name__}_history"][-1]
            if history_dict[f"{performance_func.__name__}_for_{cost_func.__name__}_history"] else None
            for performance_func in performance_funcs
        }
        final_theta = rez.x
        final_thetas[cost_func.__name__] = final_theta
        final_ansatz = create_variational_circuit(
            n_qubits, final_theta, entanglement_type=entanglement, depth=depth
        )
        final_statevector = Statevector(final_ansatz)
        final_statevectors[cost_func.__name__] = final_statevector
        final_ansatzes[cost_func.__name__] = final_ansatz

    # Prepare final results for return
    results = {
        "final_statevectors": final_statevectors,
        "history_dict": history_dict,
        "final_ansatzes": final_ansatzes,
        "final_thetas": final_thetas,
        "optimization_times": optimization_times,
        "optimization_methods": optimization_methods, 
        "shots_used_dict": shots_used_dict,
    }

    return results


def plot_vqe_results(history_dict, cost_funcs, performance_funcs):
    """
    Plots the loss functions and performance functions from the VQE run.
    """
    num_cost_funcs = len(cost_funcs)
    num_performance_funcs = len(performance_funcs)
    
    # Create figure and axes
    fig, axs = plt.subplots(num_cost_funcs, num_performance_funcs + 3, 
                            figsize=(5 * (num_performance_funcs + 3), 4 * num_cost_funcs))
    fig.suptitle("VQE Optimization Results", fontsize=16)
    
    # Ensure axs is always a 2D array, even for single plots
    if num_cost_funcs == 1 and num_performance_funcs == 1:
        axs = [[axs]]  # Single plot case
    elif num_cost_funcs == 1:
        axs = [axs]  # Single row case
    elif num_performance_funcs == 1:
        axs = [[ax] for ax in axs]  # Single column case

    for j, cost_func in enumerate(cost_funcs):
        cost_func_name = cost_func.__name__

        # Plot Cost History
        cost_history = history_dict.get(f"{cost_func_name}_cost_history", [])
        axs[j][0].plot(cost_history, label="Total Cost")
        axs[j][0].set_yscale("log")
        axs[j][0].set_title(f"{cost_func_name.replace('_', ' ').title()} - Total Cost")
        axs[j][0].set_xlabel("Iteration")
        axs[j][0].set_ylabel("Cost Value")
        axs[j][0].legend()

        # Plot Overlap History
        overlap_history = history_dict.get(f"{cost_func_name}_overlap_history", [])
        axs[j][1].plot(overlap_history, label="Overlap Term")
        axs[j][1].set_yscale("log")
        axs[j][1].set_title(f"{cost_func_name.replace('_', ' ').title()} - Overlap Term")
        axs[j][1].set_xlabel("Iteration")
        axs[j][1].set_ylabel("Overlap Value")
        axs[j][1].legend()

        # Plot Proba Lost History
        proba_lost_history = history_dict.get(f"{cost_func_name}_proba_lost_history", [])
        axs[j][2].plot(proba_lost_history, label="Proba Lost")
        axs[j][2].set_yscale("log")
        axs[j][2].set_title(f"{cost_func_name.replace('_', ' ').title()} - Proba Lost")
        axs[j][2].set_xlabel("Iteration")
        axs[j][2].set_ylabel("Proba Lost Value")
        axs[j][2].legend()

        # Plot the performance histories
        for i, performance_func in enumerate(performance_funcs):
            performance_func_name = performance_func.__name__
            performance_history = history_dict.get(f"{performance_func_name}_for_{cost_func_name}_history", [])

            # Make performance measure names more user-friendly
            if 'overlap' in performance_func_name:
                display_name = "Overlap"
            elif 'distance' in performance_func_name:
                display_name = "Distance"
            else:
                display_name = performance_func_name.replace('_', ' ').title()

            axs[j][i + 3].plot(performance_history, label=display_name)
            axs[j][i + 3].set_yscale("log")
            axs[j][i + 3].set_title(f"{display_name} with {cost_func_name.replace('_', ' ').title()}")
            axs[j][i + 3].set_xlabel("Iteration")
            axs[j][i + 3].set_ylabel("Performance Measure")
            axs[j][i + 3].legend()

    # Add labels to columns
    column_labels = ['Total Cost', 'Overlap Term', 'Proba Lost'] + [func.__name__.replace('_', ' ').title() for func in performance_funcs]
    for ax, col in zip(axs[0], column_labels):
        ax.annotate(col, xy=(0.5, 1.2), xytext=(0, 5),
                    textcoords='offset points',
                    ha='center', va='baseline', fontsize=12, weight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95], pad=1.0)
    plt.show()
