import numpy as np
from ioh import ProblemType
from population import Population
from individual import Individual
import random

class SingleObjective:
    """
    Design:
    Population based EA
    Tournament selection 
    Crossover + Mutation 
    Elitism
    Feasibility repair
    Track best ever
    """
    def evolution(
        self, 
        problem: ProblemType, 
        pop_size: int = 20,
        eval_budget: int = 10000,
        tournament_size: int = 3,
        elitism_size: int = 2
    ) -> Individual:
       
        n = problem.meta_data.n_variables
        population = Population(pop_size, n, problem)

        evaluations = pop_size

        best_ever = population.getBest()
        generation = 0

        while evaluations < eval_budget:
            generation += 1
            offspring = []

            while len(offspring) < pop_size and evaluations < eval_budget:

                parent1 = population.tournament_Selection(k=tournament_size)
                parent2 = population.tournament_Selection(k=tournament_size)

                child1, child2 = population.performCrossover(parent1, parent2)

                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()

                child1.evaluate(problem)
                evaluations += 1
                if evaluations >= eval_budget:
                    offspring.append(child1)
                    break

                child2.evaluate(problem)
                evaluations += 1

                offspring.extend([child1, child2])

            population.individuals = self._elitist_selection(
                population.individuals, offspring, pop_size, elitism_size
            )

            current_best = population.getBest()
            if current_best.fitness > best_ever.fitness:
                best_ever = current_best

            # if generation % 250 == 0 or evaluations >= eval_budget:
            #     print(f"Generation {generation}, Evaluations {evaluations}: Best = {best_ever.fitness:.2f}")

        return best_ever

    def _elitist_selection(
        self, 
        parents: list[Individual], 
        offspring: list[Individual], 
        pop_size: int, 
        elitism_size: int
    ) -> list[Individual]:
        combined = parents + offspring
        combined.sort(key=lambda ind: ind.fitness, reverse=True)
        return combined[:pop_size]



