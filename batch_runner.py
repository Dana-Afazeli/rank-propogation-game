#!/usr/bin/env python3
"""
Batch runner for the empirical study of rank-propagation dynamics.
Runs 36,000 simulations as specified in run_matrix.yaml.
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
from itertools import product
from pathlib import Path
try:
    from typing import Dict, Any, Tuple
except ImportError:
    # For older Python versions
    Dict = dict
    Any = object
    Tuple = tuple

import numpy as np
import pandas as pd
import yaml

from footrule_sync.graph_utils import erdos_renyi_connected
from footrule_sync.dynamics import run_sync_dynamics, run_async_dynamics, potential
from footrule_sync.metrics import (
    plurality_winner, consensus_fraction, disagreement_components,
    winner_flipped, footrule_shift
)


def load_config(config_path):
    """Load the run matrix configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_max_steps(expression, n, m):
    """Parse max_steps expression like '4*n*(m-1)' or 'inf'."""
    if expression == "inf":
        return float('inf')
    # Replace variables and evaluate
    expr = expression.replace('n', str(n)).replace('m', str(m))
    return int(eval(expr))


def run_single_simulation(params):
    """Run a single simulation with given parameters."""
    n, m, p, mode, rep, base_seed = params
    
    # Generate deterministic seed for this specific run
    seed = base_seed + hash((n, m, p, mode, rep)) % (2**31)
    rng = np.random.default_rng(seed)
    
    # Create graph (try multiple times to get connected graph)
    max_attempts = 100
    for _ in range(max_attempts):
        try:
            G, _ = erdos_renyi_connected(n, p, rng)
            break
        except ValueError:
            continue
    else:
        # If we can't get a connected graph, create empty result
        return {
            'n': n, 'm': m, 'p': p, 'mode': mode, 'rep': rep, 'seed': seed,
            'T_stop': 0, 'converged': 0, 'cycle_len': 0,
            'phi_0': 0, 'phi_f': 0, 'k_components': n,
            'winner_flip': 0, 'winner_item': 0, 'consensus_frac': 1.0/m,
            'delta_FR': 0.0
        }
    
    # Generate initial random permutations
    initial_perms = np.array([rng.permutation(m) for _ in range(n)], dtype=int)
    initial_phi = potential(initial_perms, G)
    
    # Set max steps based on mode
    if mode == "sync":
        max_steps = parse_max_steps("4*n*(m-1)", n, m)
    else:  # async
        max_steps = 10000  # practical limit for async
    
    # Run dynamics
    if mode == "sync":
        result = run_sync_dynamics(G, initial_perms, max_steps=max_steps, self_inclusive=False)
    else:  # async
        result = run_async_dynamics(G, initial_perms, max_steps=max_steps, 
                                  self_inclusive=False, rng=rng)
    
    final_perms = result["perms"]
    final_phi = potential(final_perms, G)
    
    # Compute all required metrics
    winner_item = plurality_winner(final_perms)
    consensus_frac = consensus_fraction(final_perms)
    k_components = disagreement_components(final_perms, G)
    winner_flip = int(winner_flipped(initial_perms, final_perms))
    delta_FR = footrule_shift(initial_perms, final_perms)
    
    return {
        'n': n, 'm': m, 'p': p, 'mode': mode, 'rep': rep, 'seed': seed,
        'T_stop': result["steps_run"],
        'converged': int(result["converged"]),
        'cycle_len': result.get("cycle_length", 0),
        'phi_0': initial_phi,
        'phi_f': final_phi,
        'k_components': k_components,
        'winner_flip': winner_flip,
        'winner_item': winner_item,
        'consensus_frac': consensus_frac,
        'delta_FR': delta_FR
    }


def generate_run_parameters(config):
    """Generate all parameter combinations for the run matrix."""
    n_values = config['n']
    m_values = config['m']
    p_values = config['p']
    modes = config['mode']
    reps = config['reps']
    base_seed = config['base_seed']
    
    for n, m, p, mode in product(n_values, m_values, p_values, modes):
        for rep in range(reps):
            yield (n, m, p, mode, rep, base_seed)


def save_result(result, outdir):
    """Save a single result to the consolidated results file."""
    filepath = outdir / "all_results.csv"
    
    # Create CSV with header if it doesn't exist
    file_exists = filepath.exists()
    
    with open(filepath, 'a', newline='') as f:
        fieldnames = ['n', 'm', 'p', 'mode', 'rep', 'seed', 'T_stop', 'converged', 
                     'cycle_len', 'phi_0', 'phi_f', 'k_components', 'winner_flip', 
                     'winner_item', 'consensus_frac', 'delta_FR']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def worker_function(params_and_outdir):
    """Worker function for multiprocessing."""
    params, outdir = params_and_outdir
    try:
        result = run_single_simulation(params)
        save_result(result, outdir)
        return f"Completed: n={params[0]}, m={params[1]}, p={params[2]}, mode={params[3]}, rep={params[4]}"
    except Exception as e:
        return f"Error: {params}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Run the empirical study batch")
    parser.add_argument("--config", required=True, help="Path to run_matrix.yaml")
    parser.add_argument("--outdir", required=True, help="Output directory for results")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--resume", action="store_true", help="Resume from existing results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Generate all parameter combinations
    all_params = list(generate_run_parameters(config))
    total_runs = len(all_params)
    
    print(f"Total simulations to run: {total_runs}")
    print(f"Using {args.jobs} parallel processes")
    
    # Filter out already completed runs if resuming
    if args.resume:
        consolidated_file = outdir / "all_results.csv"
        if consolidated_file.exists():
            existing_df = pd.read_csv(consolidated_file)
            completed_runs = set()
            for _, row in existing_df.iterrows():
                completed_runs.add((row['n'], row['m'], row['p'], row['mode'], row['rep']))
            
            remaining_params = []
            for params in all_params:
                n, m, p, mode, rep, _ = params
                if (n, m, p, mode, rep) not in completed_runs:
                    remaining_params.append(params)
            print(f"Resuming: {len(remaining_params)} runs remaining")
            all_params = remaining_params
        else:
            print("No existing consolidated file found, running all simulations")
    
    if not all_params:
        print("All simulations already completed!")
        return
    
    # Prepare arguments for worker processes
    worker_args = [(params, outdir) for params in all_params]
    
    # Run simulations in parallel
    if args.jobs == 1:
        # Sequential execution for debugging
        for i, arg in enumerate(worker_args):
            result = worker_function(arg)
            print(f"[{i+1}/{len(worker_args)}] {result}")
    else:
        # Parallel execution
        with mp.Pool(args.jobs) as pool:
            for i, result in enumerate(pool.imap_unordered(worker_function, worker_args)):
                print(f"[{i+1}/{len(worker_args)}] {result}")
    
    print("Batch run completed!")


if __name__ == "__main__":
    main()