import random

class Individual:
    def __init__(self, chromosome, fitness):
        self.chromosome = chromosome
        self.fitness = fitness

class GAwLL:
    # Constants
    POPULATION_SIZE=100
    TOURNAMENT_SIZE=3
    CROSSOVER_RATE=0.4

    TAU_RESET_GENERATIONS=50
    TAU_RESET_EPSILON=1.0e-10

    # Constructor
    def __init__(self, *, fitness_function, chrom_size, mutation_probability, max_generations):
        self.fitness_function = fitness_function
        self.chrom_size = chrom_size
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations

    # Methods
    def run(self, seed):
        random.seed(seed)

        self.initiatePopulation()
        # self.print_statistics()

        last_change_generation = 0
        last_change_highest_fitness = 0

        for generation in range(self.max_generations):
            new_population = []

            fittest_individual = self.get_fittest_individual()
            # print(fittest_individual.fitness,fittest_individual.chromosome)
            new_population.append(fittest_individual) # elitism: preserves fittest individual

            if(fittest_individual.fitness > last_change_highest_fitness + self.TAU_RESET_EPSILON):
                last_change_generation = generation
                last_change_highest_fitness = fittest_individual.fitness

            if generation - last_change_generation > self.TAU_RESET_GENERATIONS:
                last_change_generation = generation
                self.initiatePopulation(fittest_individual=fittest_individual)

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

            self.population = new_population
            # self.print_statistics()

    def initiatePopulation(self, fittest_individual = None):
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

        print(f'Highest fitness is {fittest_individual.fitness}, average fitness is {average_fitness} across {len(self.population)} individuals')

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
            if self.population[individual_rand].fitness > self.population[individual_chosen].fitness:
                individual_chosen = individual_rand
        
        return self.population[individual_chosen].chromosome

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
