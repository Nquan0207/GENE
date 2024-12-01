import value
import TSP
import test_case
import random
if __name__=="__main__":
    correct_path = 0
    for i in range(value.TESTS):
        # random.seed(i)
        distance_matrix = test_case.random_testcase[i]
        gene = TSP.genetic_algorithm(distance_matrix)

        brute_force = TSP.brute_force_algorithm(distance_matrix)

        if(gene == brute_force):
            correct_path+=1
    print(f"Percentage of correct path found by genetic algorithm: {(correct_path/value.TESTS) * 100}%")