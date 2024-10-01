import random
import numpy as np

POPULATION_SIZE = 100
MUTATION_RATE = 0.1
INT_MAX = 999999
# Distance matrix: distance_matrix[i][j] is the distance from city i to city j
distance_matrix = np.array([
        [0, 2, INT_MAX, 12, 5],
        [2, 0, 4, 8, INT_MAX],
        [INT_MAX, 4, 0, 3, 3],
        [12, 8, 3, 0, 10],
        [5, INT_MAX, 3, 10, 0],
    ])

class Individual(object):
    ''' Class representing individual in the population (a tour) '''

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def create_gnome(self):
        ''' Create a random chromosome (tour) by shuffling cities '''
        gnome = list(range(len(distance_matrix)))
        random.shuffle(gnome)
        return gnome

    def mate(self, partner):
        ''' Perform crossover and produce offspring '''
        child_chromosome = []

        # Select a subset of cities from parent 1
        start = random.randint(0, len(self.chromosome) - 1)
        end = random.randint(start, len(self.chromosome) - 1)
        child_chromosome[start:end] = self.chromosome[start:end]

        # Fill the rest from parent 2, maintaining order
        for city in partner.chromosome:
            if city not in child_chromosome:
                child_chromosome.append(city)

        return Individual(child_chromosome)

    def mutate(self):
        ''' Mutate the chromosome by swapping two cities '''
        idx1 = random.randint(0, len(self.chromosome) - 1)
        idx2 = random.randint(0, len(self.chromosome) - 1)
        self.chromosome[idx1], self.chromosome[idx2] = self.chromosome[idx2], self.chromosome[idx1]

    def calculate_fitness(self):
        ''' Calculate fitness as the total distance of the tour '''
        total_distance = 0
        for i in range(len(self.chromosome) - 1):
            city1 = self.chromosome[i]
            city2 = self.chromosome[i + 1]
            total_distance += distance_matrix[city1][city2]

        # Add the distance to return to the starting city
        total_distance += distance_matrix[self.chromosome[-1]][self.chromosome[0]]
        return total_distance


def genetic_algorithm():
    # Initialize population
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]
    generation = 1
    count = 5
    while count > 0:
        # Sort population by fitness (lower is better)
        population = sorted(population, key=lambda x: x.fitness)
        for i in range(len(population)):
            print(f"ID_tour: {i} | Tour: {population[i].chromosome} | Fitness: {population[i].fitness}")
        # Print the best solution in the population
        print(
            f"Generation: {generation} | Best Fitness: {population[0].fitness:.2f} | Tour: {population[0].chromosome}")

        # If the best fitness is small enough, stop the algorithm
        #if population[0].fitness < 65:
         #   break

        # Create a new generation
        new_generation = []

        # Elitism: Carry over the top 10% of the population
        elite_size = int(0.1 * POPULATION_SIZE)
        new_generation.extend(population[:elite_size])

        # Crossover: Generate the rest of the population by mating
        for _ in range(POPULATION_SIZE - elite_size):
            parent1 = random.choice(population[:50])  # Select from the top 50%
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            child.mutate()  # Mutate the child
            new_generation.append(child)

        # Replace the old population with the new one
        population = new_generation
        generation += 1
        count -= 1

if __name__ == '__main__':
    genetic_algorithm()
