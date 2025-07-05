import json
import os
import numpy as np
import copy
import random
from tqdm import tqdm

# Import the main module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")):
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

# Import but avoid global execution
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"), "r") as f:
    code = compile(f.read(), "main.py", "exec")
    main_vars = {}
    exec(code, main_vars)

# Setup default parameters
default_params = {
    "Y": 1.5,
    "V_Y": 0.2,
    "V_X": 0.2, 
    "V_THETA": 0.3,
    "THETA": 0.15,
    "X": 0.2,
    "Y_WIND": 0.2,
    "X_BOUNDARY": 1.0,
    "MIN_THETA": 0.05
}

# Define parameter ranges to explore
param_ranges = {
    "Y": np.arange(0.1, 3, 0.25),
    "V_Y": np.arange(0.05, 1.5, 0.25),
    "V_X": np.arange(0.05, 1.5, 0.25),
    "V_THETA": np.arange(0.05, 1.5, 0.25),
    "THETA": np.arange(0.05, 1.5, 0.25),
    "X": np.arange(0.05, 1, 0.05),
    "Y_WIND": np.arange(0.1, 1.7, 0.1),
    "X_BOUNDARY": np.arange(0.5, 2.5, 0.25),
    "MIN_THETA": np.arange(0.01, 1, 0.02)
}

def calculate_total_combinations():
    """Calculate total number of parameter combinations"""
    total = 1
    for param_values in param_ranges.values():
        total *= len(param_values)
    return total

def run_simulation(params, num_episodes=100):
    """Run simulation with given parameters and return accuracy"""
    # Apply parameters to main module
    for key, value in params.items():
        main_vars[key] = value
    
    success_count = 0
    for _ in range(num_episodes):
        _, success = main_vars["simulate"](steps=1000, policy=main_vars["reactive_agent"])
        success_count += 1 if success else 0
    
    accuracy = success_count / num_episodes * 100
    return accuracy

def generate_random_parameter_set():
    """Generate a random parameter set from the defined ranges"""
    params = copy.deepcopy(default_params)
    for key, values in param_ranges.items():
        params[key] = random.choice(values)
    return params

def optimize_parameters_randomly(num_samples=1000, output_file="best_parameters_random.json"):
    """Test random parameter combinations without storing all combinations in memory"""
    best_accuracy = 0
    best_params = copy.deepcopy(default_params)
    results_log = []
    
    total_combinations = calculate_total_combinations()
    print(f"Total possible combinations: {total_combinations}")
    print(f"Testing {num_samples} random parameter combinations...")
    
    # Set to track which combinations we've already tried
    tested_params_hash = set()
    
    # Progress bar
    with tqdm(total=num_samples) as pbar:
        tested_count = 0
        while tested_count < num_samples:
            # Generate random parameter set
            params = generate_random_parameter_set()
            
            # Convert to hashable format to check if we've tested it
            params_tuple = tuple(sorted(params.items()))
            if params_tuple in tested_params_hash:
                continue  # Skip duplicates
            
            tested_params_hash.add(params_tuple)
            
            # Test parameters
            accuracy = run_simulation(params)
            
            # Save result
            param_result = {
                "params": copy.deepcopy(params),
                "accuracy": accuracy
            }
            results_log.append(param_result)
            
            # Update best if improved
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = copy.deepcopy(params)
                
                # Save intermediate results
                with open(output_file, "w") as f:
                    json.dump({"best_params": best_params, "best_accuracy": best_accuracy}, f, indent=2)
                
                print(f"New best: {best_accuracy:.2f}% with parameters: {best_params}")
            
            tested_count += 1
            pbar.update(1)
    
    # Save final results
    with open(output_file, "w") as f:
        json.dump({
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "results_log": results_log
        }, f, indent=2)
    
    return best_params, best_accuracy

if __name__ == "__main__":
    # Ensure environment doesn't render during optimization
    main_vars["RENDER_MODE"] = None
    # Reduce episodes for faster testing during optimization
    main_vars["EPISODES"] = 1000
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Calculate total possible combinations
    total = calculate_total_combinations()
    print(f"Total possible parameter combinations: {total:,}")
    
    # Determine approach based on combination size
    if total > 1_000_000:
        print("Very large parameter space detected. Using random sampling approach.")
        # Test a reasonable number of random combinations
        num_samples = min(total, 10000)
        best_params, best_accuracy = optimize_parameters_randomly(num_samples=num_samples)
    else:
        # If parameter space is manageable, import and use the full shuffled optimizer
        from optimize_parameters import optimize_parameters
        best_params, best_accuracy = optimize_parameters(shuffle=True)
    
    print(f"Optimization complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best parameters: {best_params}")
