# run_exercise1.py
"""
Exercise 1: Single-Objective Approaches with data collection for plotting
"""

import os
import shutil
import random
import numpy as np
import pandas as pd
import ioh
import subprocess

from optimisationAlgorithms import ea11, RLS
from genetic_algorithm import run_ga
from gsemo_runner import run_gsemo
from multi_runner import run_multi
from single_runner import run_single

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RUNS = 30
BUDGET = 10_000
ROOT = "data"
FOLDER = "exercise3"
RESULTS_FILE = "results_summary.csv"

PROBLEM_IDS = {
    "MaxCoverage":     [2100, 2101, 2102, 2103],
    "MaxInfluence":    [2200, 2201, 2202, 2203],
    "PackWhileTravel": [2300, 2301, 2302],
}

ALGORITHMS = {
    "RLS": RLS,
    "EA11": ea11,
    "GA": run_ga,
    "GSEMO": run_gsemo,
    "MULTI": run_multi,
    "SINGLE": run_single
}

# ============================================================================
# DATA COLLECTION
# ============================================================================

class DataCollector:
    """Collect evaluation data for plotting"""
    def __init__(self):
        self.data = []
    
    def record(self, algorithm, problem_id, run_id, evaluations, fitness):
        self.data.append({
            'algorithm': algorithm,
            'problem_id': problem_id,
            'run_id': run_id,
            'evaluations': evaluations,
            'fitness': fitness
        })
    
    def save(self, filepath):
        df = pd.DataFrame(self.data)
        df.to_csv(filepath, index=False)
        print(f"Saved {len(self.data)} data points to {filepath}")

# Global collector
collector = DataCollector()

# ============================================================================
# ALGORITHM WRAPPER WITH TRACKING
# ============================================================================

def run_with_tracking(alg_fn, problem, budget, alg_name, problem_id, run_id):
    """
    Run algorithm and track progress at specific evaluation points
    """
    tracking_points = [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    
    # Wrapper to track evaluations
    class TrackedProblem:
        def __init__(self, original_problem):
            self.problem = original_problem
            self.eval_count = 0
            self.best_fitness = float('-inf')
            self.track_idx = 0
            
            # Explicitly expose IOH attributes
            self.meta_data = original_problem.meta_data
            try:
                self.meta = original_problem.meta
            except AttributeError:
                self.meta = original_problem.meta_data
            
        def __call__(self, x):
            fitness = self.problem(x)
            self.eval_count += 1
            
            # Update best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
            
            # Record at tracking points
            if (self.track_idx < len(tracking_points) and 
                self.eval_count >= tracking_points[self.track_idx]):
                collector.record(alg_name, problem_id, run_id, 
                               self.eval_count, self.best_fitness)
                self.track_idx += 1
            
            return fitness
        
        # Pass through other attributes as fallback
        def __getattr__(self, name):
            return getattr(self.problem, name)
    
    # Wrap the problem
    tracked_prob = TrackedProblem(problem)
    
    # Run algorithm with wrapped problem
    try:
        best_fitness, _ = alg_fn(tracked_prob, budget=budget)
    except TypeError:
        best_fitness, _ = alg_fn(tracked_prob, budget)
    
    # Record final if not already recorded
    if tracked_prob.track_idx < len(tracking_points):
        collector.record(alg_name, problem_id, run_id, budget, tracked_prob.best_fitness)
    
    return tracked_prob.best_fitness

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Clean previous data
    full_path = os.path.join(ROOT, FOLDER)
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path, exist_ok=True)
    
    print("=" * 70)
    print("COMP SCI 3316/7316 - Assignment 3 - Exercise 1")
    print("=" * 70)
    print(f"Runs: {N_RUNS} | Budget: {BUDGET:,} | Algorithms: {len(ALGORITHMS)}")
    print("=" * 70)
    
    # Run experiments
    for problem_group, problem_ids in PROBLEM_IDS.items():
        print(f"\n{problem_group}:")
        
        for problem_id in problem_ids:
            print(f"  Problem {problem_id}:", end=" ")
            
            # Generate seeds
            base_seed = np.random.SeedSequence(42 + problem_id)
            seeds = [int(s) for s in base_seed.generate_state(N_RUNS)]
            
            for alg_name, alg_fn in ALGORITHMS.items():
                results = []
                for run_id, seed in enumerate(seeds):
                    random.seed(seed)
                    np.random.seed(seed)
                    
                    problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)
                    print(f"Running {alg_name}...", end="", flush=True)
                    fitness = run_with_tracking(alg_fn, problem, BUDGET, alg_name, problem_id, run_id)
                    print("done.", flush=True)
                    results.append(fitness)
                
                mean = np.mean(results)
                std = np.std(results)
                print(f"{alg_name}:{mean:.1f}±{std:.1f}", end=" ")
            
            print()
    
    # Save all collected data
    results_path = os.path.join(full_path, RESULTS_FILE)
    collector.save(results_path)
    
    print(f"\n{'='*70}")
    print("COMPLETE! Running plotting script next:")
    print("=" * 70)

    subprocess.run(["python3", "plot_exercise1.py"])

if __name__ == "__main__":
    main()