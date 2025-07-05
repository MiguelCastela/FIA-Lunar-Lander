import json
import os
import numpy as np
import copy
import random
import time

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# GET MAIN.PY
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

# Define parameter spaces for optimization algorithms
# Format: [min, max] for each parameter
param_bounds = {
    "Y": [0.1, 3.0],
    "V_Y": [0.05, 1.5],
    "V_X": [0.05, 1.5],
    "V_THETA": [0.05, 1.5],
    "THETA": [0.05, 1.5],
    "X": [0.05, 1.0],
    "Y_WIND": [0.1, 1.7],
    "X_BOUNDARY": [0.5, 2.5],
    "MIN_THETA": [0.01, 1.0]
}

# Create a space for Bayesian optimization
space = [
    Real(param_bounds["Y"][0], param_bounds["Y"][1], name="Y"),
    Real(param_bounds["V_Y"][0], param_bounds["V_Y"][1], name="V_Y"),
    Real(param_bounds["V_X"][0], param_bounds["V_X"][1], name="V_X"),
    Real(param_bounds["V_THETA"][0], param_bounds["V_THETA"][1], name="V_THETA"),
    Real(param_bounds["THETA"][0], param_bounds["THETA"][1], name="THETA"),
    Real(param_bounds["X"][0], param_bounds["X"][1], name="X"),
    Real(param_bounds["Y_WIND"][0], param_bounds["Y_WIND"][1], name="Y_WIND"),
    Real(param_bounds["X_BOUNDARY"][0], param_bounds["X_BOUNDARY"][1], name="X_BOUNDARY"),
    Real(param_bounds["MIN_THETA"][0], param_bounds["MIN_THETA"][1], name="MIN_THETA")
]

# List of parameter names to maintain order
param_names = ["Y", "V_Y", "V_X", "V_THETA", "THETA", "X", "Y_WIND", "X_BOUNDARY", "MIN_THETA"]


def run_simulation(params, num_episodes=1000):
    # Apply parameters to main module
    for key, value in params.items():
        main_vars[key] = value
        
    print(f"""
          Running simulation with parameters:
            Y = {params["Y"]}
            V_Y = {params["V_Y"]}   
            V_X = {params["V_X"]}
            V_THETA = {params["V_THETA"]}
            THETA = {params["THETA"]}
            X = {params["X"]}
            Y_WIND = {params["Y_WIND"]}
            X_BOUNDARY = {params["X_BOUNDARY"]}
            MIN_THETA = {params["MIN_THETA"]}
          """)
    
    success_count = 0
    for _ in range(num_episodes):
        _, success = main_vars["simulate"](steps=1000000, policy=main_vars["reactive_agent"])
        success_count += 1 if success else 0
    
    accuracy = success_count / num_episodes * 100
    return accuracy


@use_named_args(space)
def objective_function(**params):
    # Convert params to dict format
    param_dict = {}
    for name in param_names:
        param_dict[name] = params[name]
    
    # Always run 1000 episodes to ensure parameters are good
    accuracy = run_simulation(param_dict, num_episodes=1000)
    
    # We minimize the negative accuracy (to maximize accuracy)
    return -accuracy


def bayesian_optimization(n_calls=50, output_file="ml_optimized_parameters.json"):
    print(f"Starting Bayesian optimization with {n_calls} iterations...")
    
    # Set up results tracking
    results_log = []
    best_accuracy = 0
    best_params = copy.deepcopy(default_params)
    
    # Use Gaussian Process based optimization
    start_time = time.time()
    result = gp_minimize(
        objective_function,
        space,
        n_calls=n_calls,
        verbose=True,
        noise=0.1,  # To handle noisy evaluations
        n_random_starts=10  # Initial random points before using surrogate model
    )
    end_time = time.time()
    
    # Process results
    opt_params = {}
    for i, name in enumerate(param_names):
        opt_params[name] = result.x[i]
    
    # Get the best accuracy (negate the minimized objective)
    best_accuracy = -result.fun
    
    # Create an optimization trace for analysis
    for i, (x, y) in enumerate(zip(result.x_iters, result.func_vals)):
        params_dict = {name: x[j] for j, name in enumerate(param_names)}
        accuracy = -y  # Negate since we're minimizing negative accuracy
        
        iteration_result = {
            "iteration": i,
            "params": params_dict,
            "accuracy": accuracy
        }
        results_log.append(iteration_result)
        
        # Update best if improved
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = copy.deepcopy(params_dict)
            
            # Log improvements
            print(f"New best: {best_accuracy:.2f}% with parameters: {best_params}")
    
    # Save final results
    optimization_result = {
        "best_params": best_params,
        "best_accuracy": best_accuracy,
        "results_log": results_log,
        "optimization_time": end_time - start_time,
        "n_iterations": n_calls
    }
    
    with open(output_file, "w") as f:
        json.dump(optimization_result, f, indent=2)
    
    print(f"Optimization complete! Results saved to {output_file}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Best parameters: {best_params}")
    print(f"Optimization took {end_time - start_time:.2f} seconds")
    
    return best_params, best_accuracy, result


def validate_best_params(params, num_episodes=1000):
    """Validate the best parameters with more episodes"""
    print(f"Validating best parameters with {num_episodes} episodes...")
    accuracy = run_simulation(params, num_episodes=num_episodes)
    print(f"Validation accuracy: {accuracy:.2f}%")
    return accuracy


def compare_with_random_search(n_iterations=50, output_file="optimization_comparison.json"):
    """Compare Bayesian optimization with random search"""
    print("Running comparison between Bayesian optimization and Random search...")
    
    # Setup tracking variables
    random_results = []
    random_best_accuracy = 0
    random_best_params = copy.deepcopy(default_params)
    random_start_time = time.time()
    
    # Generate parameter sets for random search
    for i in range(n_iterations):
        # Generate random parameters
        params = {}
        for name in param_names:
            params[name] = np.random.uniform(param_bounds[name][0], param_bounds[name][1])
        
        # Always evaluate with 1000 episodes
        accuracy = run_simulation(params, num_episodes=1000)
        
        # Track results
        result = {
            "iteration": i,
            "params": params,
            "accuracy": accuracy
        }
        random_results.append(result)
        
        # Update best
        if accuracy > random_best_accuracy:
            random_best_accuracy = accuracy
            random_best_params = copy.deepcopy(params)
            print(f"Random search new best: {random_best_accuracy:.2f}%")
    
    random_end_time = time.time()
    random_time = random_end_time - random_start_time
    
    # Run Bayesian optimization
    bayesian_start_time = time.time()
    best_params, best_accuracy, result = bayesian_optimization(n_calls=n_iterations)
    bayesian_end_time = time.time()
    bayesian_time = bayesian_end_time - bayesian_start_time
    
    # Compile comparison results
    comparison = {
        "bayesian_optimization": {
            "best_params": best_params,
            "best_accuracy": best_accuracy,
            "time_taken": bayesian_time,
            "n_iterations": n_iterations
        },
        "random_search": {
            "best_params": random_best_params,
            "best_accuracy": random_best_accuracy,
            "time_taken": random_time,
            "n_iterations": n_iterations,
            "results": random_results
        }
    }
    
    # Save comparison
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("\n===== OPTIMIZATION COMPARISON =====")
    print(f"Random Search Best Accuracy: {random_best_accuracy:.2f}%")
    print(f"Bayesian Optimization Best Accuracy: {best_accuracy:.2f}%")
    print(f"Random Search Time: {random_time:.2f} seconds")
    print(f"Bayesian Optimization Time: {bayesian_time:.2f} seconds")
    print("===================================\n")
    
    return comparison


if __name__ == "__main__":
    # Ensure environment doesn't render during optimization
    main_vars["RENDER_MODE"] = None
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="ML-based parameter optimization for Lunar Lander")
    parser.add_argument("--mode", choices=["optimize", "validate", "compare"], default="optimize", 
                        help="Mode to run the optimizer in")
    parser.add_argument("--iterations", type=int, default=50, 
                        help="Number of iterations for optimization")
    parser.add_argument("--episodes", type=int, default=100, 
                        help="Number of episodes per evaluation")
    args = parser.parse_args()
    
    # Load existing best parameters for reference
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_optimized_parameters.json"), "r") as f:
            existing_best = json.load(f)
            print(f"Existing best accuracy: {existing_best.get('best_accuracy', 0)}%")
    except FileNotFoundError:
        print("No existing best parameters found.")
    
    # Run in selected mode
    if args.mode == "optimize":
        # Run Bayesian optimization
        best_params, best_accuracy, _ = bayesian_optimization(n_calls=args.iterations, 
                                                          output_file="ml_optimized_parameters.json")
        
        # Always validate with 1000 episodes
        validate_best_params(best_params, num_episodes=1000)
        
    elif args.mode == "validate":
        # Load and validate existing ML optimized parameters
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_parameters_random.json"), "r") as f:
                ml_best = json.load(f)
                validate_best_params(ml_best["best_params"], num_episodes=1000)
        except FileNotFoundError:
            print("No ML optimized parameters found. Run in 'optimize' mode first.")
            
    elif args.mode == "compare":
        # Run comparison between Bayesian optimization and random search
        compare_with_random_search(n_iterations=args.iterations, 
                                  output_file="optimization_comparison.json")
