# Wrapper for running MultiObjective & SingleObjective functions
from multiObjective import MultiObjective
from singleObjective import SingleObjective

def run_single_objective(problem, budget=10000):
    so = SingleObjective()
    individual = so.evolution(problem=problem,pop_size=50,eval_budget=budget)
    return individual.fitness, individual.chromosome

def run_multi_objective(problem, budget=10000):
    mo = MultiObjective()
    individuals = mo.evolution(problem=problem, pop_size=50, evaluation_budget=budget)
    #for index, individual in enumerate(individuals):
    #    print(f"{index}: fitness {individual.fitness}")
    return individuals[-1].fitness, individuals[-1].chromosome