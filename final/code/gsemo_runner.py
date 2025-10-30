# gsemo_wrapper.py
from gsemo import GSEMO

def run_gsemo(problem, budget=10000):
    """
    Integrates GSEMO with run_exercise1's tracking system.
    """
    algo = GSEMO(problem_id=problem.meta_data.problem_id, budget=budget)
    
    # IMPORTANT: use the tracked problem wrapper for logging
    algo.problem = problem
    algo.n = problem.meta_data.n_variables
    
    archive, fitness = algo.run()
    best_f1 = max(f[0] for f in fitness.values())
    return best_f1, None