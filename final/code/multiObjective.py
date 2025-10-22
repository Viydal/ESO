import numpy as np
from ioh import ProblemType
from population import Population
from individual import Individual
import time
import random

class MultiObjective:

    # Multi-objective evolutionary algorithm with island model
    @staticmethod
    def evolution(problem: ProblemType, pop_size: int = 10, generation_count: int = 10000) -> list[Individual]:
        n = problem.meta_data.n_variables

        # Create two seperate islands following the island model
        pop1_size = pop_size // 2
        pop2_size = pop_size - (pop1_size)
        pop1 = Population(pop1_size, n, problem)
        pop2 = Population(pop2_size, n, problem)

        for generation in range(generation_count):
            for i in range(len(pop1.individuals) // 2):
                parent1 = pop1.pareto_tournament_selection(k=3)
                parent2 = pop1.pareto_tournament_selection(k=3)

                child1, child2 = pop1.performCrossover(parent1, parent2)

                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()

                child1.evaluate(problem)
                child2.evaluate(problem)

                pop1.individuals.extend([child1, child2])
                pop1.sort_and_clean()
            
            for i in range(len(pop2.individuals) // 2):
                parent1 = pop2.pareto_tournament_selection(k=3)
                parent2 = pop2.pareto_tournament_selection(k=3)

                child1, child2 = pop2.performCrossover(parent1, parent2)

                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()

                child1.evaluate(problem)
                child2.evaluate(problem)

                pop2.individuals.extend([child1, child2])
                pop2.sort_and_clean()

            # Perform migration between islands
            if generation % 250 == 0 and generation != 0:
                # Print current best fitness and cost
                best_fitness_pop1 = max(ind.fitness for ind in pop1.individuals)
                best_cost_pop1 = min(ind.cost for ind in pop1.individuals)

                best_fitness_pop2 = max(ind.fitness for ind in pop2.individuals)
                best_cost_pop2 = min(ind.cost for ind in pop2.individuals)

                print(f"Gen {generation}: Pop1 -> Fitness: {best_fitness_pop1}, Cost: {best_cost_pop1}")
                print(f"Gen {generation}: Pop2 -> Fitness: {best_fitness_pop2}, Cost: {best_cost_pop2}")
                
                index1 = random.randint(0, pop1_size - 1)
                index2 = random.randint(0, pop2_size - 1)

                pop1.individuals[index1], pop2.individuals[index2] = pop2.individuals[index2], pop1.individuals[index1]

                
            
