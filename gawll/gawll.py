import random, copy, cachetools
from cachetools import LRUCache


class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness


class eVIGEdge:
    def __init__(self, vertex, weight):
        self.vertex = vertex
        self.weight = weight


class eVIGNode:
    def __init__(self):
        self.edges = []

    def add_or_replace_edge(self, vertex, weight):
        found = False

        for edge in self.edges:
            if edge.vertex == vertex:
                found = True
                edge.weight = weight

        if not found:
            self.edges.append(eVIGEdge(vertex, weight))


class eVIG:
    def __init__(self, degree):
        self.degree = degree

        self._edge_frequency = [[0 for _ in range(degree)] for _ in range(degree)]
        self._edge_weight_sum = [[0 for _ in range(degree)] for _ in range(degree)]
        self._nodes = [eVIGNode() for _ in range(degree)]

    def add_edge(self, a, b, w):
        self._edge_weight_sum[a][b] += w
        if self._edge_frequency[a][b] > 0:
            w = self._edge_weight_sum[a][b] / self._edge_frequency[a][b]
        self._edge_frequency[a][b] += 1

        self._edge_weight_sum[b][a] = self._edge_weight_sum[a][b]
        self._edge_frequency[b][a] = self._edge_frequency[a][b]

        self._nodes[a].add_or_replace_edge(b, w)
        self._nodes[b].add_or_replace_edge(a, w)

    def adjacency_matrix(self):
        adjacency = [[0 for _ in range(self.degree)] for _ in range(self.degree)]
        
        for i in range(self.degree):
            for j in range(self.degree):
                if self._edge_frequency[i][j] != 0:
                    adjacency[i][j] = self._edge_weight_sum[i][j] / self._edge_frequency[i][j]
        
        return adjacency

    def print(self):
        for index, node in enumerate(self._nodes):
            print(f"  x_{index}:")
            for edge in node.edges:
                print(f"    x_{edge.vertex} ({edge.weight})")


class Statistics:
    def __init__(self):
        self.best_individual_per_run = []
        self.evig_per_run = []

        self.initial_bfi_per_run = []
        self.initial_mean_fitness_per_run = []


class GAwLL:
    # Constants
    POPULATION_SIZE = 100
    TOURNAMENT_SIZE = 3
    CROSSOVER_RATE = 0.4

    TAU_RESET_GENERATIONS = 50

    EPSILON = 1.0e-10

    FITNESS_FUNCTION_CACHE_LIMIT = 10e4

    # Constructor
    def __init__(
        self,
        *,
        fitness_function,
        chrom_size,
        mutation_probability,
        max_generations,
        linkage_learning=True,
    ):
        cache = LRUCache(maxsize=self.FITNESS_FUNCTION_CACHE_LIMIT)

        self.fitness_function = lambda chromosome: cache.get(
            tuple(chromosome),
            cache.setdefault(tuple(chromosome), fitness_function(chromosome)),
        )

        self.chrom_size = chrom_size
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations
        self.linkage_learning = linkage_learning

        self.statistics = Statistics()

    # Methods
    def run(self, seed):
        random.seed(seed)

        self.initialize_population()
        self.e_vig = eVIG(self.chrom_size)

        self.statistics.initial_bfi_per_run.append(self.get_fittest_individual())
        self.statistics.initial_mean_fitness_per_run.append(self.get_average_fitness())

        last_change_generation = 0
        last_change_highest_fitness = 0

        for generation in range(self.max_generations):
            fittest_individual = self.get_fittest_individual()

            if fittest_individual.fitness > last_change_highest_fitness + self.EPSILON:
                last_change_generation = generation
                last_change_highest_fitness = fittest_individual.fitness

            if generation - last_change_generation > self.TAU_RESET_GENERATIONS:
                last_change_generation = generation
                self.initialize_population(fittest_individual=fittest_individual)

            if self.linkage_learning:
                self.population = self.generation_ll(fittest_individual)
            else:
                self.population = self.generation(fittest_individual)

        self.statistics.best_individual_per_run.append(self.get_fittest_individual())
        self.statistics.evig_per_run.append(self.e_vig)

    def generation(self, fittest_individual):
        new_population = []

        # elitism: preserves fittest individual
        new_population.append(self.generate_individual(fittest_individual.chromosome.copy()))

        while (diff := len(self.population) - len(new_population)) > 0:
            offspring1 = parent1 = self.selection()

            if diff > 1:
                offspring2 = parent2 = self.selection()

                if random.random() < self.CROSSOVER_RATE:
                    offspring1, offspring2 = self.uniform_crossover(parent1, parent2)

                self.mutation(offspring1)
                self.mutation(offspring2)

                new_population.append(self.generate_individual(offspring1))
                new_population.append(self.generate_individual(offspring2))
            else:
                self.mutation(offspring1)
                new_population.append(self.generate_individual(offspring1))

        return new_population

    def generation_ll(self, fittest_individual):
        new_population = []

        # elitism: preserves fittest individual
        new_population.append(self.generate_individual(fittest_individual.chromosome.copy()))

        # number of individuals for standard crossover and mutation
        nc = int(self.POPULATION_SIZE * self.CROSSOVER_RATE)

        while (diff := len(self.population) - len(new_population)) > 0:
            offspring1 = parent1 = self.selection()

            if len(new_population) < nc:
                offspring2 = parent2 = self.selection()

                if random.random() < self.CROSSOVER_RATE:
                    offspring1, offspring2 = self.uniform_crossover(parent1, parent2)

                self.mutation(offspring1)
                self.mutation(offspring2)

                new_population.append(self.generate_individual(offspring1))
                new_population.append(self.generate_individual(offspring2))
            elif diff > 2:
                offspring1, offspring2, offspring3 = self.mutation_ll(parent1)

                new_population.append(self.generate_individual(offspring1))
                new_population.append(self.generate_individual(offspring2))
                new_population.append(self.generate_individual(offspring3))
            else:
                self.mutation(offspring1)
                new_population.append(self.generate_individual(offspring1))

        return new_population

    def initialize_population(self, fittest_individual=None):
        population = []

        for _ in range(self.POPULATION_SIZE):
            chromosome = [bool(random.randint(0, 1)) for _ in range(self.chrom_size)]
            population.append(self.generate_individual(chromosome))

        if fittest_individual is not None:
            population[0] = fittest_individual

        self.population = population

    def generate_individual(self, chromosome):
        fitness = self.fitness_function(chromosome)
        return Individual(chromosome, fitness)

    def print_population(self):
        for individual in self.population:
            print(individual.fitness, individual.chromosome)

    def print_statistics(self):
        fittest_individual = self.get_fittest_individual()
        average_fitness = self.get_average_fitness()

        print(
            f"Highest fitness is {fittest_individual.fitness}, average fitness is {average_fitness} across {len(self.population)} individuals"
        )

    def get_average_fitness(self):
        if not self.population:
            return 0

        total_fitness = sum(individual.fitness for individual in self.population)
        average = total_fitness / len(self.population)

        return average

    def get_fittest_individual(self):
        if not self.population:
            return None

        fittest_individual = self.population[0]
        for individual in self.population[1:]:
            if individual.fitness > fittest_individual.fitness:
                fittest_individual = individual

        return fittest_individual

    def selection(self):
        individual_chosen = random.randint(0, self.POPULATION_SIZE - 1)

        for _ in range(self.TOURNAMENT_SIZE):
            individual_rand = random.randint(0, self.POPULATION_SIZE - 1)
            if (
                self.population[individual_rand].fitness
                > self.population[individual_chosen].fitness
            ):
                individual_chosen = individual_rand

        return self.population[individual_chosen].chromosome.copy()

    def uniform_crossover(self, parent1, parent2):
        chromosome1 = []
        chromosome2 = []

        for gene in range(self.chrom_size):
            if parent1[gene] == parent2[gene]:
                perform_crossover = False
            else:
                perform_crossover = random.choice([False, True])

            if perform_crossover:
                chromosome1.append(parent2[gene])
                chromosome2.append(parent1[gene])
            else:
                chromosome1.append(parent1[gene])
                chromosome2.append(parent2[gene])

        return chromosome1, chromosome2

    def mutation(self, offspring):
        for gene in offspring:
            if random.random() < self.mutation_probability:
                offspring[gene] = not offspring[gene]

    def mutation_ll(self, parent):
        xg = parent.copy()
        xh = parent.copy()
        xgh = parent.copy()

        xg_mutation = random.randint(0, self.chrom_size - 1)

        xh_mutation = random.randint(0, self.chrom_size - 2)
        if xh_mutation >= xg_mutation:
            xh_mutation += 1

        fx = self.fitness_function(parent)

        xg[xg_mutation] = not xg[xg_mutation]
        fxg = self.fitness_function(xg)

        xh[xh_mutation] = not xh[xh_mutation]
        fxh = self.fitness_function(xh)

        xgh[xg_mutation] = not xgh[xg_mutation]
        xgh[xh_mutation] = not xgh[xh_mutation]
        fxgh = self.fitness_function(xgh)

        df = abs(fxgh - fxh - fxg + fx)
        if df > self.EPSILON:
            self.e_vig.add_edge(xg_mutation, xh_mutation, df)

        return xg, xh, xgh
