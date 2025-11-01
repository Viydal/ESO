# import random
# import numpy as np
# import math
# from ioh import ProblemType


# class Individual:
#     chromosome: list[int]
#     fitness: float
#     def __init__(self, n: int, chromosome: list[int] | None = None):
#         if chromosome is None:
#             self.chromosome = list(np.random.randint(2, size=n))
#         else:
#             self.chromosome = chromosome
#         self.fitness = -math.inf
#         self.cost = -math.inf

#     def evaluate(self, problem: ProblemType):
#         self.fitness = problem(self.chromosome)
#         self.cost = sum(self.chromosome)
#         return self.fitness, self.cost
    
#     # Flip a single random bit
#     def singleRandomMutation(self):
#         # Select a random index in the chromosome
#         idx = random.randint(0, len(self.chromosome) - 1)
#         # Flip the bit at the selected index
#         self.chromosome[idx] = 1 - self.chromosome[idx]
#         return self
    
#     # Flip all bits with prob 1/n
#     def flipAllBitsMutation(self):
#         n = len(self.chromosome)
#         for i in range(n):
#             if np.random.rand() < 1.0 / n:
#                 #Flip the bit
#                 self.chromosome[i] = 1 - self.chromosome[i]
# individual.py
import random
import numpy as np
import math
from ioh import ProblemType
from typing import List, Tuple

class Individual:
    chromosome: List[int]
    fitness: float
    cost: float

    def __init__(self, n: int, chromosome: List[int] | None = None):
        if chromosome is None:
            # Produce a plain Python list[int] (no numpy scalars)
            self.chromosome = list(map(int, np.random.randint(2, size=n)))
        else:
            self.chromosome = list(map(int, chromosome))
        self.fitness = -math.inf  # assuming maximisation
        self.cost = -math.inf

    # --- NEW: GA wrapper expects this ---
    @property
    def representation(self) -> List[int]:
        """Return a defensive copy of the genome as list[int]."""
        return list(self.chromosome)

    # Optional: handy when you need a clone
    def copy(self) -> "Individual":
        clone = Individual(len(self.chromosome), self.chromosome)
        clone.fitness = self.fitness
        clone.cost = self.cost
        return clone

    def evaluate(self, problem: ProblemType) -> Tuple[float, float]:
        """Evaluate via IOH problem(x) to ensure logging & budget accounting."""
        # IOH problems are callable: y = problem(x)
        f = float(problem(self.chromosome))
        self.fitness = f
        # 'cost' here is arbitrary; keep if you use it, or remove entirely
        self.cost = float(sum(self.chromosome))
        return self.fitness, self.cost

    # Flip a single random bit (in-place) and RETURN self for chaining
    def singleRandomMutation(self) -> "Individual":
        idx = random.randint(0, len(self.chromosome) - 1)  # inclusive bounds OK
        self.chromosome[idx] = 1 - self.chromosome[idx]
        return self

    # Flip each bit with prob 1/n (in-place) and RETURN self for chaining
    def flipAllBitsMutation(self) -> "Individual":
        n = len(self.chromosome)

        for i in range(n):
            if np.random.rand() < 1.0 / n:
                self.chromosome[i] = 1 - self.chromosome[i]
        return self
