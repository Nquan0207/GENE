import matplotlib.pyplot as plt
import test_case
import TSP

if __name__=="__main__":
    distance_matrix = test_case.random_testcase[1]
    gene, best_fitness, generations = TSP.genetic_algorithm(distance_matrix)

    best_solution, fitness_progress = gene, best_fitness

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_progress, label="Best Fitness Over Generations", color="blue")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Total Distance)")
    plt.title("Convergence of Genetic Algorithm in TSP")
    plt.legend()
    plt.show()

    print(f"Best solution found: {best_solution}")
    print(f"Shortest distance: {min(fitness_progress)}")
    print(f"Number of generations: {generations}")

