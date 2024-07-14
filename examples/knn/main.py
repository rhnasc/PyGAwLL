import sys
import os

# Adiciona o diretório raiz do projeto ao PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gawll')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../util')))

from gawll import GAwLL
from util import Util

def main():

    (chrom_size,X_trainset, d_trainset,X_testset, d_testset) = Util.read_dataset("zoo")

    # Total number of runs
    total_runs=1
    # Mutation Rate
    mutation_probability=1.0/chrom_size
    # Maximum number of generations
    max_generations=100

    fitness_function = lambda array: sum(array) / len(array) if array else 0.0

    instance = GAwLL(fitness_function=fitness_function, chrom_size=chrom_size,mutation_probability=mutation_probability,max_generations=max_generations)


    for i in range(total_runs):
        instance.run(i+1)


if __name__ == '__main__':
    main()
