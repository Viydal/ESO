from ioh import get_problem, ProblemType, ProblemClass, logger
import ioh
import sys
import numpy as np
import optimisationAlgorithms
from genetic import GeneticAlgorithm
from gsemo import GSEMO

def random_search(func, iterations, budget = None) -> tuple[float, list[int]]:
    # budget of each run: 50n^2
    if budget is None:
        budget = int(func.meta_data.n_variables * func.meta_data.n_variables * 50)

    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y
    #print(optimum)
    # 10 independent runs for each algorithm on each problem.
    f_opt: float = sys.float_info.min
    x_opt: list[int] = list(np.random.randint(2, size = func.meta_data.n_variables))
    for r in range(iterations):
        f_opt = sys.float_info.min
        x_opt = list(np.random.randint(2, size = func.meta_data.n_variables))
        for i in range(budget):
            x: list[int] = list(np.random.randint(2, size = func.meta_data.n_variables))
            f: float = func(x)
            if f > f_opt:
                f_opt = f
                x_opt = x
            if f_opt >= optimum:
                break
        func.reset()
    return f_opt, x_opt


if __name__ == '__main__':

    algorithms = {
    # "RLS": optimisationAlgorithms.RLS,
    # "(1+1)EA": optimisationAlgorithms.ea11,
    # "GA": GeneticAlgorithm(pop_size=30, generations=10000),
    "GSEMO": GSEMO
    }


    problems = []
    num_dim = 100
    f = [
        2100, 2101, 2102, 2103,
        2200, 2201, 2202, 2203,
        2300, 2301, 2302
    ]

    for i in f:
        problems.append(get_problem(fid = i, dimension = num_dim, instance = 1, problem_class = ProblemClass.GRAPH))

    from ioh import logger

BUDGET = 10000
REPEATS = 30

for alg_name, alg_func in algorithms.items():
    print(f"=== Running {alg_name} ===")

    for problem in problems:
        print(f"Problem {problem.meta_data.problem_id}: {problem.meta_data.name}")

        # Create a logger 
        l = logger.Analyzer(
            root="data",
            folder_name=f"run_{alg_name}",
            algorithm_name=alg_name,
            algorithm_info="Submodular optimisation experiment"
        )
        problem.attach_logger(l)

        for r in range(REPEATS):
            print(f"Run {r+1}/{REPEATS}")
            problem.reset()  # Reset between runs

            # Call algorithm
            if alg_name == "GA":
                alg_func.run(problem, budget=BUDGET) 
            elif alg_name == "GSEMO":
                # Create a GSEMO instance for this problem
                gsemo = GSEMO(problem_id=problem.meta_data.problem_id, budget=BUDGET)
                # Override its problem with the one from IOH logger (same instance)
                gsemo.problem = problem
                gsemo.run()
            else:
                alg_func(problem, BUDGET)

        problem.detach_logger()  # Finalize logging
