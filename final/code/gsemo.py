import random
import ioh

class GSEMO:
    def __init__(self, problem_id, budget=10000, seed=None):
        # Load the given problem instance
        self.problem = ioh.get_problem(problem_id, problem_class=ioh.ProblemClass.GRAPH)
        
        # Number of variables (length of bitstring)
        self.n = self.problem.meta_data.n_variables
        
        # Number of fitness evaluations
        self.budget = budget
        
        # Number of evaluations performed
        self.evaluations = 0
        
        # Archive stores unique bitstrings that are non-dominated
        self.archive = set()
        
        # Fitness dictionary maps bitstring -> (f1, f2)
        # f1 = main objective value (e.g. coverage, influence, etc.)
        # f2 = cost (number of selected nodes)
        self.fitness = dict()


    def evaluate(self, bitstring):
        # Convert string -> int vector only for IOH
        x = []
        i = 0
        while i < self.n:
            if bitstring[i] == '1':
                x.append(1)
            else:
                x.append(0)
            i += 1
        # main objective
        f1 = self.problem(x)
        # cost (# selected nodes)
        f2 = sum(x)
        self.fitness[bitstring] = (f1, f2)
        self.evaluations += 1
        return f1, f2

    def dominates(self, a, b):
        fa = self.fitness[a]
        fb = self.fitness[b]
        if fa[0] >= fb[0] and fa[1] <= fb[1]:
            if fa[0] > fb[0] or fa[1] < fb[1]:
                return True
        return False

    def mutate(self, parent):
        i = 0
        result = ''
        while i < self.n:
            if random.random() < 1.0 / self.n:
                if parent[i] == '1':
                    result += '0'
                else:
                    result += '1'
            else:
                result += parent[i]
            i += 1
        return result

    def random_from_set(self, s):
        # Pick a random element from a set without converting to list/tuple
        target_index = random.randint(0, len(s) - 1)
        j = 0
        for elem in s:
            if j == target_index:
                return elem
            j += 1
        return elem  # fallback

    def run(self):
        # Initialize random bitstring of length n
        x = ''
        i = 0
        while i < self.n:
            if random.random() < 0.5:
                x += '1'
            else:
                x += '0'
            i += 1
        self.evaluate(x)
        self.archive.add(x)

        while self.evaluations < self.budget:
            parent = self.random_from_set(self.archive)
            offspring = self.mutate(parent)
            self.evaluate(offspring)

            # Check dominance
            dominated = set()
            nondominated = True
            for y in self.archive:
                if self.dominates(y, offspring):
                    nondominated = False
                    break
                if self.dominates(offspring, y):
                    dominated.add(y)

            if nondominated:
                for d in dominated:
                    self.archive.remove(d)
                self.archive.add(offspring)

        return self.archive, self.fitness


# -------------------------------
# Main runner for assignment tasks
# -------------------------------
def main():
    instances = [
        2100, 2101, 2102, 2103,
        2200, 2201, 2202, 2203,
        2300, 2301, 2302
    ]

    for pid in instances:
        print(f"\n=== Running GSEMO_Pure on instance {pid} ===")
        algo = GSEMO(problem_id=pid, budget=10000, seed=42)
        archive, fitness = algo.run()

        print(f"Total evaluations: {algo.evaluations}")
        print(f"Non-dominated solutions: {len(archive)}")

        # Print up to 3 representative solutions
        count = 0
        for bitstring in archive:
            f = fitness[bitstring]
            print(f"  f1={f[0]:.3f}, f2={f[1]}")
            count += 1
            if count >= 3:
                break

        print("...")

if __name__ == "__main__":
    main()
