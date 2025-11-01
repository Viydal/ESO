from individual import Individual
import random
import math
import numpy as np
import copy


class Population:
    individuals: list[Individual]
    # Initialise a list of individuals as a population
    def __init__(self, size, n, problem):
        self.individuals = [Individual(n) for _ in range(size)]
        self.size = size
        for ind in self.individuals:
            ind.evaluate(problem)

    # Return the best individual within the population
    def getBest(self) -> Individual | None:
        if not self.individuals:
            return None

        return max(self.individuals, key=lambda ind: ind.fitness)
        
    # Generalised function to centralise the execution of a crossover
    def performCrossover(self, parent1, parent2, crossover_probability=0.8) -> tuple[Individual, Individual]:
        # Should crossover occur, if not return parents without modification
        random_number: float = random.random()
        if random_number > crossover_probability:
            return Individual(len(parent1.chromosome), parent1.chromosome.copy()), \
            Individual(len(parent2.chromosome), parent2.chromosome.copy())

        # If crossover should occur, perform crossover and return children
        child1: Individual | None = None
        child2: Individual | None = None
        
        child1, child2 = self.uniform_Crossover(parent1, parent2)
        
        # if either child is None then the crossover was incomplete, return parents
        if child1 is None or child2 is None:
            return parent1, parent2

        return child1, child2

    # tournament selection for parent selection
    def tournament_Selection(self, k: int = 3) -> Individual:
        # Randomly select k individuals from the population
        tournament: list[Individual] = random.sample(self.individuals, k)
        # Get the individual with the lowest cost
        winner: Individual = max(tournament, key=lambda ind: ind.fitness)
        return winner

    # Informal tournament selection for parent selection
    # This method selects two parents using tournament selection
    # Not sure if right
    def informal_tournament_selection(self, k: int = 3) -> tuple[Individual,Individual]:
        parent1: Individual = self.tournament_Selection(k)
        parent2: Individual = self.tournament_Selection(k)
        return parent1, parent2
    
    def uniform_Crossover(self, parent1: Individual, parent2) -> tuple[Individual, Individual]:
        length: int = len(parent1.chromosome)
        mask: list[int] = list(np.random.randint(0, 2, size=length))
        child1: Individual = copy.deepcopy(parent1)
        child2: Individual = copy.deepcopy(parent2)
        for i in range(length):
            if mask[i] == 1:
                child1.chromosome[i] = parent2.chromosome[i]
                child2.chromosome[i] = parent1.chromosome[i]
        return child1, child2
    
    def pareto_tournament_selection(self, k=3):
        candidates = random.sample(self.individuals, k)
        winner = candidates[0]
        for individual in candidates:
            if (self.dominates(individual, winner)):
                winner = individual
            elif not self.dominates(winner, individual):
                # Neither dominates the other
                winner = random.choice([winner, individual])
        
        return winner

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
    
    def sort_and_clean(self):
        fronts = self.pareto_fronts(self.individuals)

        new_population = []

        for front in fronts:
            if len(front) + len(new_population) <= self.size:
                new_population.extend(front)
            else:
                remaining_spots = self.size - len(new_population)

                if remaining_spots > 0:
                    crowding_distances = self.calculate_crowding_distance(front)
                    front_sorted = sorted(front, key=lambda x: crowding_distances[x], reverse=True)
                    new_population.extend(front_sorted[:remaining_spots])
            
                break

        self.individuals = new_population

    def pareto_fronts(self, individuals: list[Individual]):
        fronts = []
        remaining = individuals.copy()

        while (remaining):
            to_remove = []
            next_front = []

            for individual in remaining:
                dominated = False
                for other in remaining:
                    if other is not individual and self.dominates(other, individual):
                        dominated = True
                        break
                if not dominated:
                    next_front.append(individual)
                    to_remove.append(individual)
            
            if not next_front:
                break

            fronts.append(next_front)

            for individual in to_remove:
                remaining.remove(individual)

        return fronts

    def calculate_crowding_distance(self, individuals: list[Individual]):
        if len(individuals) <= 2:
            result = {}
            for individual in individuals:
                result[individual] = float('inf')
            return result
            
        objectives = []
        for individual in individuals:
            fitness_value = individual.fitness
            cost_value = -individual.cost
            objectives.append([fitness_value, cost_value])
        
        objectives = np.array(objectives)
        n = len(individuals)

        distances = np.zeros(n)

        for obj_idx in range(2):
            sorted_indices = np.argsort(objectives[:, obj_idx])

            obj_values = objectives[sorted_indices, obj_idx]

            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')

            obj_range = obj_values[-1] - obj_values[0]

            if obj_range == 0:
                continue

            for i in range(1, n - 1):
                individual_index = sorted_indices[i]

                next_value = obj_values[i + 1]
                prev_value = obj_values[i - 1]

                distance_contribution = (next_value - prev_value) / obj_range
                distances[individual_index] += distance_contribution
        
        result = {}
        for i in range(n):
            individual = individuals[i]
            distance = distances[i]
            result[individual] = distance
        
        return result