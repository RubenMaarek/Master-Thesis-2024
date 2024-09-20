# Master Thesis Project: Variational Quantum Algorithms Applied to Quantitative Finance and Options Pricing

This repository contains the code for my Master Thesis on **Variational Quantum Algorithms Applied to Quantitative Finance and Options Pricing**. The project leverages quantum computing to tackle problems in finance, such as options pricing and risk modeling.

## Project Structure

- `lib/`: Core functions to define the model and implement the Variational Quantum Eigensolver (VQE) algorithm.
- `Pipeline/pipeline_scripts`: Scripts that define the pipeline for executing experiments.
- `Pipeline/dvc`: Configuration files to run the pipeline locally or on an HPC, allowing for batch job execution.
- `result_app/`: Visualization folder using Streamlit to display results on a local web interface.

## How to Run

1. **Set Up the Pipeline**:  
    In `run_pipeline.py`, specify your `base_model_dir`. The `params.yaml` file defines the number of iterations and qubit values for the runs. To start the pipeline, run:
    ```bash
    python run_pipeline.py --n-qubits $n_qubits --iteration $iteration --depth $depth

2. **Update the Visualization Paths**:
    In the `result_app/pages` folder, update the model paths to point to the newly created directories from the pipeline.
3. **Launch Visualization**:
    Start the Streamlit app by running: 
    ```bash
    streamlit run result_app/main.py

Navigate to the provided local URL to visualize the results from the pipeline.





