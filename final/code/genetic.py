from population import Population
from individual import Individual
import numpy as np
from ioh import ProblemType

class GeneticAlgorithm:
    pop_size: int
    generations: int
    
    def __init__(self, pop_size=20, generations=1000):
        self.pop_size = pop_size
        self.generations = generations

    def run(self, problem: ProblemType, budget: int = None) -> Individual | None:
        n: int = problem.meta_data.n_variables
        pop: Population = Population(self.pop_size, n, problem)
        best: Individual | None = pop.getBest()
        if best is None: 
            return None

        # Track evaluations
        evaluations_used = self.pop_size  # Initial population
        
        # Calculate max generations based on budget if provided
        if budget:
            # Each generation uses pop_size * 2 evaluations
            max_generations = min(self.generations, (budget - evaluations_used) // (self.pop_size * 2))
        else:
            max_generations = self.generations
        
        for gen in range(max_generations):
            new_inds: list[Individual] = []
            
            for i in range(self.pop_size):
                # Check budget
                if budget and evaluations_used >= budget:
                    return best
                    
                # tournament selection
                parent1, parent2 = pop.informal_tournament_selection(k=3)
                child1, child2 = pop.performCrossover(parent1, parent2)
                child1.flipAllBitsMutation()
                child2.flipAllBitsMutation()
                child1.evaluate(problem)
                child2.evaluate(problem)
                new_inds.extend([child1, child2])
                
                if budget:
                    evaluations_used += 2

            # elitism: keep best of old + new
            elite: Individual | None = pop.getBest()
            if elite is None: 
                return best
                
            pop.individuals = sorted(new_inds,
                                   key=lambda ind: ind.fitness, reverse=True)[:self.pop_size]
            popBest: Individual | None = pop.getBest()
            if popBest is None: 
                return best
                
            if elite.fitness > popBest.fitness:
                pop.individuals[-1] = elite

            if popBest.fitness > best.fitness:
                best = pop.getBest()
                if best is None: 
                    return best

        return best