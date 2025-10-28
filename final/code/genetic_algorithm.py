from genetic import GeneticAlgorithm
from ioh import ProblemType

def run_ga(problem: ProblemType, budget: int = 10000) -> tuple[float, list[int]]:
    pop_size = 20
    # If you evaluate an initial population of size pop_size, account for it:
    offspring_per_gen = pop_size * 2  # your comment says 2 children per pair
    generations = max(1, (budget - pop_size) // offspring_per_gen)

    ga = GeneticAlgorithm(pop_size=pop_size, generations=generations)
    best_individual = ga.run(problem)   # must be an Individual with .fitness set via problem(x)
    if best_individual is None:
        return float("-inf"), []
    return float(best_individual.fitness), best_individual.representation
