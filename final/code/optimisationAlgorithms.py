from typing import Sequence
from ioh import get_problem, ProblemType, ProblemClass, logger
import sys
import numpy as np
import random, math

# EA(1+1) Algorithm
def ea11(problem: ProblemType, budget: int = 100000) -> tuple[float,list[int]]:
    # Set budget to 50n^2
    if budget is None:
        budget = int(problem.meta_data.n_variables * problem.meta_data.n_variables * 50)
    # setup Problem Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum: int = 8
    else:
        optimum: int = problem.optimum.y

    # Setup x & f
    f_opt: float = float("-inf")
    x_opt: list[int] = list(np.random.randint(2, size = problem.meta_data.n_variables))
    for _ in range(budget):
        x: list[int] = x_opt.copy()

        # Create list of probabilities for bit changes
        rand: list[int] = list(np.random.randint(len(x), size = problem.meta_data.n_variables))
        for i in range(len(x)):
            if rand[i] == 0:
                # Flip Bit
                x[i] = 1-x[i]
        f = problem(x)

        # If better than optimum update optimum
        if f > f_opt:
            f_opt = f
            x_opt = x

        # if better than problem optimum then return.
        if f_opt >= optimum:
            print(f"done in {_} iterations.")
            break

    return f_opt, x_opt

# RLS
import random, math
from ioh import ProblemType

# def RLS(problem: ProblemType, budget: int = 100000) -> tuple[float, list[int]]:
#     # 1) Get dimension from the problem (avoid hard-coding n)
#     try:
#         n = problem.meta.n_variables
#     except AttributeError:
#         n = problem.meta_data.n_variables  # fallback for older IOH versions

#     # 2) Detect optimization direction
#     try:
#         opt_type = problem.meta.optimization_type.name.lower()
#     except AttributeError:
#         opt_type = problem.meta_data.optimization_type.name.lower()
#     maximize = (opt_type == "maximization")

#     # 3) Random start and seed best with the first evaluation
#     x = [random.randint(0, 1) for _ in range(n)]
#     fx = float(problem(x))  # logs + counts against budget
#     if math.isnan(fx):
#         # If NaN, re-sample until we get a valid start (very defensive)
#         attempts = 0
#         while math.isnan(fx) and attempts < 10:
#             x = [random.randint(0, 1) for _ in range(n)]
#             fx = float(problem(x))
#             attempts += 1
#         if math.isnan(fx):
#             # give up gracefully
#             return float("-inf") if maximize else float("+inf"), x

#     best_x = x[:]
#     best_f = fx
#     evals = 1

#     # 4) Main loop (1-bit flip hill climbing)
#     while evals < budget:
#         i = random.randrange(n)
#         x[i] ^= 1
#         f_new = float(problem(x))
#         evals += 1

#         if not math.isnan(f_new):
#             improved = (f_new > best_f) if maximize else (f_new < best_f)
#             if improved:
#                 best_f = f_new
#                 best_x = x[:]
#             else:
#                 # revert if not better
#                 x[i] ^= 1
#         else:
#             # revert on NaN
#             x[i] ^= 1

#     return best_f, best_x
def RLS(problem, budget=100000):
    """Random Local Search - assumes maximization"""
    try:
        n = problem.meta_data.n_variables
    except AttributeError:
        try:
            n = problem.meta.n_variables
        except AttributeError:
            n = len(problem([0] * 50))  # fallback - test with dummy solution
            n = 50  # or get from elsewhere
    
    # Random start
    x = [random.randint(0, 1) for _ in range(n)]
    best_fitness = problem(x)
    best_solution = x[:]
    evals = 1
    
    # Main loop
    while evals < budget:
        i = random.randint(0, n - 1)
        x[i] = 1 - x[i]  # flip bit
        
        fitness = problem(x)
        evals += 1
        
        # Accept if BETTER (higher is better for maximization)
        if fitness > best_fitness:
            best_fitness = fitness
            best_solution = x[:]
        else:
            x[i] = 1 - x[i]  # revert
    
    return best_fitness, best_solution