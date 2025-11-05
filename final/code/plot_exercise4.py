# plot_exercise1.py
"""
Generate fixed-budget plots from collected data, including GSEMO
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILE = "final/doc/exercise4/Exercise4-Summary.csv"
OUTPUT_FOLDER = "final/doc/exercise4"

PROBLEM_NAMES = {
    2100: "MaxCoverage-2100", 2101: "MaxCoverage-2101",
    2102: "MaxCoverage-2102", 2103: "MaxCoverage-2103",
    2200: "MaxInfluence-2200", 2201: "MaxInfluence-2201",
    2202: "MaxInfluence-2202", 2203: "MaxInfluence-2203",
    2300: "PackWhileTravel-2300", 2301: "PackWhileTravel-2301",
    2302: "PackWhileTravel-2302",
}

def create_plot(df, problem_id):
    """Create fixed-budget plot for one problem"""
    plt.figure(figsize=(10, 6))
    
    problem_data = df[df['problem_id'] == problem_id]
    algorithms = problem_data['algorithm'].unique()
    
    colors = {
        'RLS': 'blue',
        'EA11': 'red',
        'GA': 'green',
        'GSEMO': 'purple',
        "MULTI": 'black',
        "SINGLE": 'yellow'
    }
    
    for alg in algorithms:
        alg_data = problem_data[problem_data['algorithm'] == alg]
        
        # Group by evaluation points
        grouped = alg_data.groupby('evaluations')['fitness']
        mean = grouped.mean()
        std = grouped.std()
        evals = mean.index
        
        # Plot
        plt.plot(evals, mean, label=alg, color=colors.get(alg, 'black'), linewidth=2)
        plt.fill_between(evals, mean - std, mean + std, 
                         color=colors.get(alg, 'black'), alpha=0.2)
    plt.xscale("log")
    plt.xlabel('Function Evaluations', fontsize=12)
    plt.ylabel('Fitness Value', fontsize=12)
    plt.title(f'{PROBLEM_NAMES.get(problem_id, f"Problem {problem_id}")}', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    output_path = f"{OUTPUT_FOLDER}/problem_{problem_id}_fixed_budget.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ {PROBLEM_NAMES.get(problem_id, f'Problem {problem_id}')}")

def main():
    print("=" * 70)
    print("Generating Fixed-Budget Plots including GSEMO")
    print("=" * 70)
    
    # Load data
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found!")
        print("Run 'python run_exercise1.py' first")
        return
    
    df = pd.read_csv(DATA_FILE)
    print(f"Loaded {len(df)} data points")
    
    # Generate plots for each problem
    print("\nGenerating plots:")
    for problem_id in sorted(df['problem_id'].unique()):
        create_plot(df, problem_id)
    
    print(f"\n{'='*70}")
    print(f"All plots saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
