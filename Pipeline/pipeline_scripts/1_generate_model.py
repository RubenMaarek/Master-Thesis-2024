import sys
import os
from pathlib import Path
import numpy as np
import json
import typer
import pandas as pd

# Add the root directory (VARIATIONAL-MARTINGALE-SOLVER) to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from lib.new_model import random_payoff_system_from_qubits
from lib.utils import extract_original_matrix  # Import the function to extract original matrix



def generate_model(output_path: Path = typer.Option(...), n_qubits: int = typer.Option(6), timestamp: str = typer.Option(None)):
    
    os.makedirs(output_path.parent, exist_ok=True)

    S, Pi, mus, sigmas, D, strike_prices, payoff_functions, MC_option_prices, qp_solution_vector, N, K, norm_Pi = random_payoff_system_from_qubits(n_qubits=n_qubits)
    i = np.random.randint(0, N)

    model_config = {
        "n_qubits": n_qubits,
        "N": N,
        "K": K,
        "i": i,
        "norm_Pi": norm_Pi,
        "payoff_functions": payoff_functions,
    }

    with open(output_path, 'w') as json_file:
        json.dump(model_config, json_file, indent=4)

    # Correctly save matrices
    pd.DataFrame(S).to_csv(output_path.parent / "S_matrix.csv", index=False)
    pd.DataFrame(D).to_csv(output_path.parent / "D_matrix.csv", index=False)

    # Correctly save vectors
    pd.DataFrame(Pi.reshape(-1,1)).to_csv(output_path.parent / "Pi_vector.csv", index=False, header=False)
    pd.DataFrame(mus.reshape(-1,1)).to_csv(output_path.parent / "mus_drifts.csv", index=False, header=False)
    pd.DataFrame(sigmas.reshape(-1,1)).to_csv(output_path.parent / "sigmas_volatilities.csv", index=False, header=False)
    pd.DataFrame(strike_prices.reshape(-1,1)).to_csv(output_path.parent / "strike_prices.csv", index=False, header=False)
    pd.DataFrame(qp_solution_vector.reshape(-1,1)).to_csv(output_path.parent / "qp_solution_vector.csv", index=False, header=False)
    pd.DataFrame(MC_option_prices.reshape(-1,1)).to_csv(output_path.parent / "MC_option_prices.csv", index=False, header=False)
    

    original_D = extract_original_matrix(D, N, K)
    qp_option_prices = original_D @ qp_solution_vector
    
    # Prepare a DataFrame to store all option prices
    all_option_prices_df = pd.DataFrame({
        'Index': np.arange(N),
        'MC_Option_Prices': MC_option_prices,
        'QP_Option_Prices': qp_option_prices[:N],
    })
    
    # Save the all_option_prices DataFrame to a CSV file
    all_option_prices_df.to_csv(output_path.parent / "all_option_prices.csv", index=False)

if __name__ == "__main__":
    typer.run(generate_model)