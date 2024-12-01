import value
import TSP
import time
import test_case
import random
if __name__=="__main__":
    total_generations = 0
    for i in range(value.TESTS):
        distance_matrix = test_case.random_testcase[i]
        gene, best_fitness, num_gen = TSP.genetic_algorithm(distance_matrix)
        total_generations += num_gen

    average_gen = total_generations/100
    print(f"The total generations of this variant: {total_generations}")
    print(f"The average generations of this variant: {average_gen}")



