import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

FOLDER = "final/code/data/exercise4"

subfolders = os.listdir(FOLDER)
print(subfolders)

AlgConvert = {"ea11": "EA11","GA": "GA", "RLS": "RLS", "SingleObjective": "SINGLE", "MultiObjective": "MULTI", "GSEMO": "GSEMO"}

probsToDims = {2100: 450, 2101: 450, 2102: 595, 2103: 760, 2200: 4039, 2201: 4039, 2202: 4039, 2203: 4039}

output = open(os.path.join(FOLDER,"../../../doc/exercise4/Exercise4-Summary.csv"),'w')
output.write("algorithm,problem_id,run_id,evaluations,fitness\n")
for folder in subfolders:
    algo, problem = folder[10:].split('-')
    print(algo, problem)
    if int(problem) < 2200:
        file = os.path.join(FOLDER,f"{folder}/data_f{problem}_MaxCoverage{problem}/IOHprofiler_f{problem}_DIM{probsToDims[int(problem)]}.dat")
    else:
        file = os.path.join(FOLDER,f"{folder}/data_f{problem}_MaxInfluence{problem}/IOHprofiler_f{problem}_DIM{probsToDims[int(problem)]}.dat")
    print(os.path.exists(file))
    print(file)

    content = open(file).readlines()
    index = -1
    max_fitness = None
    for line in content:
        if line.strip() == '':
            break
        if line == "evaluations raw_y\n":
            index+= 1
            max_fitness = None
        else:
            eval, fitness = line.strip().split(' ')
            fitness = float(fitness)
            if max_fitness is None:
                max_fitness = fitness
            if fitness >= max_fitness:
                output.write(f"{AlgConvert[algo]},{problem},{index},{eval},{fitness}\n")
                max_fitness = fitness
            else:
                print(eval, fitness)



