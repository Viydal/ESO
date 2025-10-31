import numpy as np
from ioh import ProblemType
from population import Population
from individual import Individual
import random

class SingleObjective:

    def evolution(
        self, 
        problem: ProblemType, 
        pop_size: int = 20,
        generation_count: int = 10000,
        tournament_size: int = 3,
        elitism_size: int = 2
    ) -> Individual:
        
        n = problem.meta_data.n_variables
        population = Population(pop_size, n, problem)
        
        for ind in population.individuals:
            if ind.fitness < 0:
                self._repair(ind, problem)
        
        best_ever = population.getBest()
        
        for generation in range(generation_count):
            offspring = []
            for _ in range(pop_size // 2):
                parent1 = population.tournament_Selection(k=tournament_size)
                parent2 = population.tournament_Selection(k=tournament_size)
                
                child1, child2 = population.performCrossover(parent1, parent2)
                
                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()
                
                child1.evaluate(problem)
                child2.evaluate(problem)
                
                if child1.fitness < 0:
                    self._repair(child1, problem)
                if child2.fitness < 0:
                    self._repair(child2, problem)
                
                offspring.extend([child1, child2])
            
            population.individuals = self._elitist_selection(
                population.individuals, offspring, pop_size, elitism_size
            )
            
            current_best = population.getBest()
            if current_best.fitness > best_ever.fitness:
                best_ever = current_best
            
            if generation % 250 == 0:
                print(f"Generation {generation}: Best = {best_ever.fitness:.2f}")
        
        return best_ever

    def _repair(self, individual: Individual, problem: ProblemType):
        while individual.fitness < 0:
            ones_indices = [i for i, bit in enumerate(individual.chromosome) if bit == 1]
            if not ones_indices:
                break
            idx = random.choice(ones_indices)
            individual.chromosome[idx] = 0
            individual.evaluate(problem)

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
