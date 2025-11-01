from multiObjective import MultiObjective
from ioh import get_problem, ProblemType, ProblemClass, logger

if __name__ == "__main__":
    problems = []
    instanceIDs = [2100, 2101, 2102, 2103, 2200, 2201, 2202, 2203, 2300, 2301, 2302]

    for id in instanceIDs:
        problems.append(get_problem(fid = id, problem_class = ProblemClass.GRAPH))

    l = logger.Analyzer(root = "data",
    folder_name = "objective",
    algorithm_name="multi-objective EA",
    algorithm_info="test of IOHExperimenter in python")
    print("multi-objective EA")

    for problem in problems:
        print("\n=============================")
        print(f"\nProblem {problem.meta_data.problem_id}: {problem.meta_data.name}\n")
        problem.attach_logger(l)
        mo = MultiObjective()
        pareto_front = mo.evolution(problem=problem, pop_size=20, evaluation_budget=10000)
        print()
        for index, individual in enumerate(pareto_front):
            print(f"individual {index + 1}: cost: {individual.cost} fitness: {individual.fitness}")
            # print(f"{individual.chromosome}")
        problem.reset()
    print(f"The problem name is: {problem.meta_data.name}")
    print(f"The problem optimum is: {problem.optimum.y}\n")