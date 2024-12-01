import random
import crossover
import mutate
import selection
import value
from individual import Individual
from itertools import permutations


####################################################################################################
def genetic_algorithm(distance_matrix):
    population = [Individual(Individual.create_gnome(), distance_matrix) for _ in range(value.POPULATION_SIZE)]
    count = value.GENERATIONS
    best_fitness = []
    best_value = value.INT_MAX
    stagnant_generation = 0
    max_stagnant_generation = 20


    while count > 0:
        # Sort population by fitness (lower is better)
        population = sorted(population, key=lambda x: x.fitness)
        for i in range(len(population)):
            print(f"ID_tour: {i} | Tour: {population[i].chromosome} | Fitness: {population[i].fitness}")
        print(
            f"Generation: {value.GENERATIONS - count + 1} | Best Fitness: {population[0].fitness:.2f} | Tour: {population[0].chromosome}")

        # Create a new generation
        new_generation = []

        # Elitism: Carry over the top 10% of the population
        elite_size = int(0.1 * value.POPULATION_SIZE)
        new_generation.extend(population[:elite_size])

        while (len(new_generation) < int(value.POPULATION_SIZE * 0.8)):
            parent1 = selection.roulette_wheel(population)
            parent2 = selection.roulette_wheel(population)
            child1, child2 = crossover.order_crossover(parent1, parent2, distance_matrix)
            if random.random() < 0.1:
                mutate.inverse_mutate(child1)
                child1.calculate_fitness(distance_matrix)
            if random.random() < 0.1:
                mutate.inverse_mutate(child2)
                child2.calculate_fitness(distance_matrix)
            if (child1.check_valid_path()):
                new_generation.append(child1)
            if (child2.check_valid_path()):
                new_generation.append(child2)

        while (len(new_generation) < value.POPULATION_SIZE):
            new_generation.append(Individual(Individual.create_gnome(), distance_matrix))
        # Replace the old population with the new one
        population = new_generation
        best_fitness.append(population[0].fitness)
        count -= 1
        current_best_fitness = population[0].fitness
        if current_best_fitness < best_value:
            best_value = current_best_fitness
            stagnant_generation = 0  # Reset stagnant counter
        else:
            stagnant_generation += 1

        # Early stopping condition
        if stagnant_generation >= max_stagnant_generation:
            print("Stopping early: No improvement for 20 consecutive generations.")
            break

    return population[0].chromosome, best_fitness, (value.GENERATIONS - count)


def brute_force_algorithm(distance_matrix):
    # m: number of cites
    # Time complexity: O(m!)
    # Space complexity: O(1)
    min = value.INT_MAX
    min_path = []
    for path in (permutations(range(0, value.CITIES))):
        distance = 0
        for i in range(len(path) - 1):
            city1 = path[i]
            city2 = path[i + 1]
            distance += distance_matrix[city1][city2]
            if distance > min:
                break

        if min > distance:
            min = distance
            min_path = path

    print(f"Best path: {min_path} | Distance: {min}")
    return min

