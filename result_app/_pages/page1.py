# result_app/pages/page1.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path

def main(selected_run_dir):
    st.title("Generated Model")

    selected_run_path = Path(selected_run_dir)
    model_path = selected_run_path / "model_config.json"

    # Load and display the model configuration
    with open(model_path, 'r') as json_file:
        model_config = json.load(json_file)
    
    st.header("Model Configuration")
    st.json(model_config)
    norm_Pi = model_config["norm_Pi"]
    

    # Load and display matrices and vectors
    df_S = pd.read_csv(selected_run_path / "S_matrix.csv")
    df_Pi = pd.read_csv(selected_run_path / "Pi_vector.csv")
    df_D = pd.read_csv(selected_run_path / "D_matrix.csv")
    df_strike_prices = pd.read_csv(selected_run_path / "strike_prices.csv")

    st.header("S Matrix")
    st.dataframe(norm_Pi * df_S)

    st.header("Pi Vector")
    st.dataframe(norm_Pi * df_Pi)

    st.header("Strike Prices")
    st.dataframe(df_strike_prices)

    try : 
        st.header("Payoff Functions")
        payoff_functions = model_config["payoff_functions"]
        for i, payoff_function in enumerate(payoff_functions):
            st.write(f"Payoff Function {i + 1}: {payoff_function}")
    except:
        st.warning("No payoff functions found in model configuration")

    st.header("D Matrix")
    st.dataframe(df_D)

if __name__ == "__main__":
    main()
