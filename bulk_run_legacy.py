import os
import sys
import pathlib
import importlib
from game import run_game
import lib.time_logger as TLOG
from tqdm import tqdm
import contextlib
import io

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

# Define directories for attacker and defender policies.
attacker_dir = os.path.join(ROOT_DIR, "policies", "attacker")
defender_dir = os.path.join(ROOT_DIR, "policies", "defender")

# Get all attacker and defender modules.
attacker_modules = get_policy_modules(attacker_dir)
defender_modules = get_policy_modules(defender_dir)

tqdm.write("Attacker modules: " + str(list(attacker_modules.keys())))
tqdm.write("Defender modules: " + str(list(defender_modules.keys())))

# Get all configuration files from the config directory.
config_dir = os.path.join(ROOT_DIR, "data/config")
config_files = [f for f in os.listdir(config_dir) if f.endswith(".yml")]

tqdm.write("Configuration files found: " + str(len(config_files)))

RESULT_PATH = os.path.join(ROOT_DIR, "data/result")

# Calculate total iterations for the progress bar.
total_iterations = len(config_files) * len(attacker_modules) * len(defender_modules)
pbar = tqdm(total=total_iterations, desc="Bulk Run", ncols=150)

# Create a dummy output to capture unwanted prints.
dummy_out = io.StringIO()

with contextlib.redirect_stdout(dummy_out):
    for config_file in config_files:
        for attacker_name, attacker_module in attacker_modules.items():
            for defender_name, defender_module in defender_modules.items():
                tqdm.write(f"\nRunning game for config '{config_file}' with attacker '{attacker_name}' vs defender '{defender_name}'...")
                # Create a new logger for this run using the config file name.
                logger = TLOG.TimeLogger(config_file, path=RESULT_PATH)
                logger.set_metadata({"attacker_strategy": attacker_name, "defender_strategy": defender_name})
                payoff, game_time, total_captures, total_tags = run_game(
                    config_file,
                    root_dir=str(ROOT_DIR),
                    attacker_strategy=attacker_module,
                    defender_strategy=defender_module,
                    logger=logger,
                    visualization=False,
                    debug=True
                )
                # Derive the log file name:
                log_filename = (
                    config_file.replace("config_", "log_")
                               .replace(".yml", f"_attacker_{attacker_name}_defender_{defender_name}.json")
                )
                logger.write_to_file(log_filename, force=True)
                tqdm.write(f"Log saved: {log_filename}")
                pbar.update(1)
        #         break
        #     break
        # break
pbar.close()
tqdm.write("Bulk run complete. Logs saved for each configuration and strategy pair.")