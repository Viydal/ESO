from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np
from run_exercise1 import run_with_tracking, DataCollector
from optimisationAlgorithms import ea11, RLS # RLS & (1+1)EA
from genetic_algorithm import run_ga # GA
from gsemo_runner import run_gsemo # GSEMO
from objective_runners import run_single_objective, run_multi_objective # Single Objective & MultiObjective

if __name__ == "__main__":
    problems = [2100,2101,2102,2103,2200,2201,2202,2203]
    algos = [["ea11",ea11], ["RLS", RLS], ["GA",run_ga], ["GSEMO", run_gsemo], ["SingleObjective", run_single_objective], ["MultiObjective",run_multi_objective]]

    #l = logger.Analyzer(root = "data",
    #folder_name="exercise4",
    #algorithm_name="default")
    if len(sys.argv) > 1:
        alg = algos[int(sys.argv[1])]
    else:
        alg = algos[0]
    
    collector = DataCollector()

    for pro in problems:
        results = []
        problem = get_problem(pro, problem_class=ProblemClass.GRAPH)
        l = logger.Analyzer(root = "ESO/final/code/data/exercise4",
        folder_name=f"exercise4-{alg[0]}-{pro}",
        algorithm_name=str(alg[0]),
        algorithm_info=str(pro))
        problem.attach_logger(l)
        for _ in range(30):
            fitness = run_with_tracking(alg[1],problem,100000,alg[0],pro,_,collector)
            results.append(fitness)
            problem.reset()
            print(f"problem: {pro}, index: {_}, algorithm: {alg[0]}")
    collector.save(f"ESO/final/code/data/Exercise4-{alg[0]}.csv")
            