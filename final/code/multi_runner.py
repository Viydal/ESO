from multiObjective import MultiObjective

def run_multi(problem, budget=10000):
    mo = MultiObjective()
    pareto_front = mo.evolution(problem=problem, pop_size=20, evaluation_budget=budget)
    return pareto_front, None