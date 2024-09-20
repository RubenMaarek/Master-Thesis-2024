# run command:  python run_pipeline.py --initialize
import subprocess
import os
import sys
import argparse
import yaml
import json
import time
from datetime import datetime
from filelock import FileLock
import psutil
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor

PARAMS_FILE = "params.yaml"
STATE_FILE = "pipeline_state.yaml"
STATE_LOCK = "pipeline_state.lock"

def initialize_log(log_file):
    with open(log_file, 'w') as f:
        f.write(f"{datetime.now()}: Starting new pipeline run\n")

def log_message(log_file, message):
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: {message}\n")

def log_resource_usage(log_file, interval=0.5):  # Reduced interval to minimize overhead
    process = psutil.Process(os.getpid())
    cpu_usage = psutil.cpu_percent(interval=interval)
    memory_usage = process.memory_info().rss / (1024 * 1024)  # In MB
    log_message(log_file, f"Current CPU usage: {cpu_usage}%")
    log_message(log_file, f"Current memory usage: {memory_usage} MB")

def log_selected_data_from_json(json_path, log_file):
    try:
        if not os.path.exists(json_path):
            log_message(log_file, f"File {json_path} does not exist.")
            return

        if os.path.getsize(json_path) == 0:
            log_message(log_file, f"File {json_path} is empty.")
            return

        with open(json_path, 'r') as f:
            data = json.load(f)

        selected_data = {key: value[-1] if isinstance(value, list) and value else value for key, value in data.items()}
        log_message(log_file, f"Final values from {json_path}:\n{json.dumps(selected_data, indent=4)}")

    except json.JSONDecodeError as json_err:
        log_message(log_file, f"Failed to parse JSON data from {json_path}: {str(json_err)}")
    except Exception as e:
        log_message(log_file, f"Failed to read or log JSON data from {json_path}: {str(e)}")

def run_command(command, log_file, cwd=None):
    log_message(log_file, f"Running command: {command}")
    try:
        start_time = time.time()
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        stdout, stderr = process.communicate()
        duration = time.time() - start_time
        log_message(log_file, f"Command completed in {duration:.2f} seconds")
        log_message(log_file, f"Standard Output:\n{stdout.decode()}")
        log_message(log_file, f"Standard Error:\n{stderr.decode()}")
        if process.returncode != 0:
            log_message(log_file, f"Error executing command: {command}")
            raise subprocess.CalledProcessError(process.returncode, command)
    except Exception as e:
        log_message(log_file, f"Exception occurred: {str(e)}")
        raise

def load_params():
    with open(PARAMS_FILE, "r") as f:
        return yaml.safe_load(f)

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return yaml.safe_load(f)
    return {"qubit_states": {}}

def save_state(state):
    with open(STATE_FILE, "w") as f:
        yaml.safe_dump(state, f)

def initialize_state():
    params = load_params()
    state = load_state()
    state.update({
        "qubit_states": {
            n_qubits: {"completed_iterations": 0}
            for n_qubits in range(params["min_qubits"], params["max_qubits"] + 1)
        }
    })
    save_state(state)

def submit_job(n_qubits, iteration, log_file):
    params = load_params()
    job_name = f"pipeline_{n_qubits}_qubits_iteration_{iteration}"
    cmd = [
        "qsub",
        "-N", job_name,
        "-l", "select=1:ncpus=8:mem=16gb",
        "-l", "walltime=02:00:00",
        "-v", f"n_qubits={n_qubits},iteration={iteration},depth={params['depth']}",
        "run_pipeline.pbs"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        print(f"Job submitted successfully: {job_id}")
        log_message(log_file, f"Submitted job {job_name} with ID {job_id}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.stderr}")
        log_message(log_file, f"Failed to submit job {job_name}: {str(e)}")
        raise

def submit_all_jobs(log_file):
    params = load_params()
    with concurrent.futures.ThreadPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
        futures = []
        for n_qubits in range(params["min_qubits"], params["max_qubits"] + 1):
            for iteration in range(1, params["iterations"] + 1):
                futures.append(executor.submit(submit_job, n_qubits, iteration, log_file))
        concurrent.futures.wait(futures)

def handle_job_completion(n_qubits, iteration, log_file):
    with FileLock(STATE_LOCK):
        state = load_state()
        params = load_params()

        state["qubit_states"][n_qubits]["completed_iterations"] += 1
        save_state(state)

        log_message(log_file, f"Completed job for {n_qubits} qubits, iteration {iteration}")
        log_message(log_file, f"Progress: {state['qubit_states'][n_qubits]['completed_iterations']}/{params['iterations']} iterations for {n_qubits} qubits")

def run_pipeline_step(step_name, script_name, args, log_file, run_dir):
    log_message(log_file, f"Step {step_name}: Running {script_name}...")
    pipeline_scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pipeline_scripts'))
    cmd = f'"{sys.executable}" "{os.path.join(pipeline_scripts_dir, script_name)}" {args}'
    run_command(cmd, log_file, cwd=run_dir)
    log_message(log_file, f"{step_name} completed successfully.")
    log_resource_usage(log_file)

def main():
    parser = argparse.ArgumentParser(description="Run quantum finance pipeline.")
    parser.add_argument("--n-qubits", type=int, help="Number of qubits to use.")
    parser.add_argument("--depth", type=int, help="Depth of the quantum circuit.")
    parser.add_argument("--run-dir", type=str, help="Directory to store the results.")
    parser.add_argument("--iteration", type=int, help="Current iteration.")
    parser.add_argument("--initialize", action="store_true", help="Initialize the pipeline state and submit all jobs.")
    args = parser.parse_args()

    params = load_params()

    if args.initialize:
        initialize_state()
        log_file = "pipeline_initialization.log"
        initialize_log(log_file)
        submit_all_jobs(log_file)
        sys.exit(0)

    n_qubits = args.n_qubits if args.n_qubits is not None else params.get("min_qubits", 2)
    depth = args.depth if args.depth is not None else params.get("depth", 1)
    iteration = args.iteration if args.iteration is not None else 1

    base_model_dir = "/hpctmp/e1124919/model"
    # base_model_dir = '/Users/rubenmaarek/Desktop/Master-Thesis-NUS/variational-martingale-solver/model'
    run_dir = args.run_dir if args.run_dir is not None else os.path.join(base_model_dir, f"run_qubits_{n_qubits}_depth_{depth}_iteration_{iteration}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    try:
        os.makedirs(run_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {run_dir}: {str(e)}")
        sys.exit(1)

    log_file = os.path.join(run_dir, 'run_pipeline.log')
    initialize_log(log_file)

    log_message(log_file, "Initial system resource usage:")
    log_resource_usage(log_file)

    try:
        # Step 1: Run generate_model.py
        run_pipeline_step("1", "1_generate_model.py", f'--output-path "{run_dir}/model_config.json" --n-qubits {n_qubits}', log_file, run_dir)

        # Run Step 2 and Step 4 concurrently, as they only depend on Step 1
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_vqe_script = executor.submit(run_pipeline_step, "2", "2_vqe_script.py", f'--model-config-path "{run_dir}/model_config.json" --output-path "{run_dir}/vqe_output.json" --depth {depth}', log_file, run_dir)
            future_vqe_for_lp_funcs_part1 = executor.submit(run_pipeline_step, "4", "4_vqe_script_for_lp_funcs.py", f'--model-config-path "{run_dir}/model_config.json" --output-path "{run_dir}/vqe_lp_output.json" --depth {depth}', log_file, run_dir)

        # Wait for Step 2 to complete before running Step 3
        future_vqe_script.result()
        run_pipeline_step("3", "3_compare_derivative_prices.py", f'--run-dir "{run_dir}"', log_file, run_dir)

        # Wait for Step 4 to complete before running Step 5
        future_vqe_for_lp_funcs_part1.result()
        run_pipeline_step("5", "4_vqe_script_for_lp_funcs.py", f'--model-config-path "{run_dir}/model_config.json" --output-path "{run_dir}/vqe_lp_output.json"', log_file, run_dir)

        handle_job_completion(n_qubits, iteration, log_file)
    except Exception as e:
        log_message(log_file, f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
