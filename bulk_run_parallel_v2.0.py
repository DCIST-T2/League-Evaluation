import os
import sys
import pathlib
import importlib
import multiprocessing
from game import run_game
import lib.time_logger as TLOG
from tqdm import tqdm
import contextlib
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import platform

# --- Import Logic (DO NOT CHANGE) ---
current_path = pathlib.Path(__file__).resolve()
ROOT_DIR = current_path.parent
sys.path.insert(0, str(ROOT_DIR))
# --- End Import Logic ---

def get_policy_modules(directory):
    """
    Returns a dictionary of module names to imported modules for all Python files
    in the given directory (ignores __init__.py and files starting with '__').
    """
    modules = {}
    for file in os.listdir(directory):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]  # Remove the .py extension
            full_module_name = f"policies.{os.path.basename(directory)}.{module_name}"
            module = importlib.import_module(full_module_name)
            modules[module_name] = module
    return modules

def run_single_game(args):
    """
    Executes a single game run and returns the log filename.
    Uses a single args parameter for better compatibility with multiprocessing.
    """
    config_file, attacker_name, defender_name = args
    try:
        # Import modules inside the function to ensure proper imports in subprocess
        full_attacker_module_name = f"policies.attacker.{attacker_name}"
        full_defender_module_name = f"policies.defender.{defender_name}"
        attacker_module = importlib.import_module(full_attacker_module_name)
        defender_module = importlib.import_module(full_defender_module_name)
        
        # Create a new logger for this run using the config file name
        RESULT_PATH = os.path.join(ROOT_DIR, "data/result")
        logger = TLOG.TimeLogger(config_file, path=RESULT_PATH)
        logger.set_metadata({"attacker_strategy": attacker_name, "defender_strategy": defender_name})
        
        # Run the game with minimal output
        payoff, game_time, total_captures, total_tags = run_game(
            config_file,
            root_dir=str(ROOT_DIR),
            attacker_strategy=attacker_module,
            defender_strategy=defender_module,
            logger=logger,
            visualization=False,
            debug=False  # Reduce stdout noise
        )
        
        # Derive the log file name
        log_filename = (
            config_file.replace("config_", "log_")
                    .replace(".yml", f"_attacker_{attacker_name}_defender_{defender_name}.json")
        )
        logger.write_to_file(log_filename, force=True)
        
        return {
            'success': True,
            'log_filename': log_filename,
            'config': config_file,
            'attacker': attacker_name,
            'defender': defender_name,
            'payoff': payoff,
            'game_time': game_time
        }
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'config': config_file,
            'attacker': attacker_name,
            'defender': defender_name
        }

def get_optimal_process_count():
    """
    Determines the optimal number of processes based on system architecture.
    For M1/M2 Macs, considers the efficiency vs performance cores.
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Check if running on Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # M1/M2 Macs typically have efficiency and performance cores
        # We want to leave some efficiency cores available for system tasks
        # For most M1/M2 Macs, a good balance is using 75% of available cores
        return max(1, int(cpu_count * 0.75))
    else:
        # For other systems, use CPU count - 1 to leave one core for system operations
        return max(1, cpu_count - 1)

def main():
    # Define directories for attacker and defender policies
    attacker_dir = os.path.join(ROOT_DIR, "policies", "attacker")
    defender_dir = os.path.join(ROOT_DIR, "policies", "defender")
    
    # Get all attacker and defender modules
    attacker_modules = get_policy_modules(attacker_dir)
    defender_modules = get_policy_modules(defender_dir)
    
    print(f"Attacker modules: {list(attacker_modules.keys())}")
    print(f"Defender modules: {list(defender_modules.keys())}")
    
    # Get all configuration files from the config directory
    config_dir = os.path.join(ROOT_DIR, "data/config")
    config_files = [f for f in os.listdir(config_dir) if f.endswith(".yml")]
    
    print(f"Configuration files found: {len(config_files)}")
    
    # Calculate total iterations for the progress bar
    total_iterations = len(config_files) * len(attacker_modules) * len(defender_modules)
    
    # Prepare all test cases
    test_cases = []
    for config_file in config_files:
        for attacker_name in attacker_modules.keys():
            for defender_name in defender_modules.keys():
                test_cases.append((config_file, attacker_name, defender_name))
    
    # Get optimal process count for this system
    num_processes = get_optimal_process_count()
    print(f"Running with {num_processes} processes on {platform.machine()} architecture")
    
    # For M1/M2 Macs, use 'spawn' method for process creation
    # This is more reliable on macOS though slightly slower to start processes
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn', force=True)
    
    # Use ProcessPoolExecutor for CPU-bound tasks
    # This bypasses the Global Interpreter Lock (GIL)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # For Apple Silicon, use 'map' for better memory management
        results = list(tqdm(
            executor.map(run_single_game, test_cases),
            total=len(test_cases),
            desc="Processing games",
            ncols=100
        ))
        
    # Process results after completion
    successful_runs = 0
    failed_runs = 0
    
    print("\nSummarizing results:")
    for result in results:
        if result['success']:
            successful_runs += 1
        else:
            failed_runs += 1
            print(f"FAILED: {result['config']} - {result['attacker']} vs {result['defender']}")
            print(f"Error: {result['error']}")
    
    print(f"\nBulk run complete. Successful runs: {successful_runs}, Failed runs: {failed_runs}")
    print(f"Logs saved in {os.path.join(ROOT_DIR, 'data/result')}")

if __name__ == "__main__":
    main()