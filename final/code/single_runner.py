from singleObjective import SingleObjective

def run_single(problem, budget=10000):
    so = SingleObjective()
    single = so.evolution(problem=problem, pop_size=20, eval_budget=budget)
    return single, None