import json
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from qiskit.quantum_info import Statevector
import typer

# Add the root directory (VARIATIONAL-MARTINGALE-SOLVER) to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from lib.alg import get_probas
from lib.config import cost_funcs

def analyze_results(run_dir: Path = typer.Option(...)):
    """
    Analyze the results of the VQE execution and store the computed option prices
    for different methods (VQE, QP, MC, LP) in a single CSV file.

    Args:
        run_dir (Path): The directory containing the results to analyze.
    """
    try:
        # Load configuration and result files
        model_config = load_json(run_dir / "model_config.json")
        N = model_config["N"]
        K = model_config["K"]

        # Load the existing all_option_prices.csv file
        all_option_prices_df = load_csv(run_dir / "all_option_prices.csv")

        # Load the final_statevectors.csv and final_statevectors_lp.csv files
        final_statevectors_df = load_csv(run_dir / "final_statevectors.csv")
        final_statevectors_lp_df = load_csv(run_dir / "final_statevectors_lp.csv")

        D_matrix = load_csv(run_dir / "D_matrix.csv").values

        # Calculate VQE option prices and add to the DataFrame
        for cost_func in cost_funcs:
            cost_func_name = cost_func.__name__
            if cost_func_name in final_statevectors_df.columns:
                statevector_data = final_statevectors_df[cost_func_name].values
                statevector = Statevector(statevector_data)
                probas_vector = get_probas(statevector)
                VQE_option_prices = D_matrix @ probas_vector
                all_option_prices_df[f'{cost_func_name}_VQE_Option_Prices'] = VQE_option_prices[:N]
            else:
                print(f"Warning: No statevector data for {cost_func_name}")

        # Calculate LP option prices and add to the DataFrame
        lp_cost_funcs = [
            "vqe_cost_function_for_dmin",
            "vqe_sampling_cost_function_for_dmin",
            "vqe_cost_function_for_dmax",
            "vqe_sampling_cost_function_for_dmax"
        ]
        lp_price_names = ["d_min", "d_min_sampling", "d_max", "d_max_sampling"]

        for cost_func, price_name in zip(lp_cost_funcs, lp_price_names):
            if cost_func in final_statevectors_lp_df.columns:
                statevector_data = final_statevectors_lp_df[cost_func].values
                statevector = Statevector(statevector_data)
                probas_vector = get_probas(statevector)
                LP_option_prices = D_matrix @ probas_vector
                all_option_prices_df[f'{price_name}_LP_Option_Prices'] = LP_option_prices[:N]
            else:
                print(f"Warning: No LP statevector data for {cost_func}")

        # Save the updated all_option_prices_df to a CSV file
        csv_path = run_dir / "all_option_prices.csv"
        all_option_prices_df.to_csv(csv_path, index=False)

        print(f"All option prices have been saved to '{csv_path}'.")
        
        # Verify that the file has been updated
        updated_df = pd.read_csv(csv_path)
        print(f"Updated CSV file contains {len(updated_df.columns)} columns.")
        print(f"Columns: {', '.join(updated_df.columns)}")

    except Exception as e:
        print(f"Error during analysis: {e}")

def load_json(file_path: Path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return {}

def load_csv(file_path: Path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading CSV file {file_path}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    typer.run(analyze_results)