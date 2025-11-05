import numpy as np
from ioh import ProblemType
from population import Population
from individual import Individual
import time
import random

class MultiObjective:

    # Multi-objective evolutionary algorithm with island model
    def evolution(self, problem: ProblemType, pop_size: int = 10, evaluation_budget: int = 10000) -> list[Individual]:
        n = problem.meta_data.n_variables

        # Create two seperate islands following the island model
        pop1_size = pop_size // 2
        pop2_size = pop_size - (pop1_size)
        pop1 = Population(pop1_size, n, problem)
        pop2 = Population(pop2_size, n, problem)

        evaluations = 0
        generations = 0
        while evaluations <= evaluation_budget:
            for i in range(len(pop1.individuals) // 2):
                parent1 = pop1.pareto_tournament_selection(k=3)
                parent2 = pop1.pareto_tournament_selection(k=3)

                child1, child2 = pop1.performCrossover(parent1, parent2)

                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()

                child1.evaluate(problem)
                child2.evaluate(problem)
                evaluations += 2
                if evaluations >= evaluation_budget:
                    break

                pop1.individuals.extend([child1, child2])
            
            for i in range(len(pop2.individuals) // 2):
                parent1 = pop2.pareto_tournament_selection(k=3)
                parent2 = pop2.pareto_tournament_selection(k=3)

                child1, child2 = pop2.performCrossover(parent1, parent2)

                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()

                child1.evaluate(problem)
                child2.evaluate(problem)
                evaluations += 2
                if evaluations >= evaluation_budget:
                    break

                pop2.individuals.extend([child1, child2])

            pop1.sort_and_clean()
            pop2.sort_and_clean()

            # Perform migration between islands
            if generations % 25 == 0:
                
                index1 = random.randint(0, pop1_size - 1)
                index2 = random.randint(0, pop2_size - 1)

                pop1.individuals[index1], pop2.individuals[index2] = pop2.individuals[index2], pop1.individuals[index1]

                # print(f"Generation {generations} complete - {evaluations} evaluations.")
            
            generations += 1

        combined_population = pop1.individuals + pop2.individuals
        return self.get_pareto_front(combined_population)
    
    def get_pareto_front(self, population: list[Individual]):
        valid_population = []
        for individual in population:
            if individual.cost != 0:
                valid_population.append(individual)

        pareto_front = []
        for ind in valid_population:
            dominated = False
            for other in valid_population:
                if (other != ind and self.dominates(other, ind)):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(ind)

        unique_front = []
        seen = set()
        for individual in pareto_front:
            key = (individual.fitness, individual.cost)
            if key not in seen:
                unique_front.append(individual)
                seen.add(key)

        unique_front.sort(key=lambda x: x.cost)
        return unique_front
    
    def dominates(self, individual1: Individual, individual2: Individual):
        if individual1.fitness < individual2.fitness:
            return False
        elif individual1.cost > individual2.cost:
            return False
        
        if individual1.fitness > individual2.fitness:
            return True
        elif individual1.cost < individual2.cost:
            return True
        
        return False