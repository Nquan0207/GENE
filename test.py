import random

def mate(somthin, partner):
    ''' Perform crossover and produce offspring '''
    child_chromosome = []

    # Select a subset of cities from parent 1
    start = random.randint(0, len(somthin) - 1)
    print(start)

    end = random.randint(start, len(somthin) - 1)
    print(end)

    child_chromosome[start:end] = somthin[start:end]

    # Fill the rest from parent 2, maintaining order
    for city in partner:
        if city not in child_chromosome:
            child_chromosome.append(city)

    return child_chromosome
def genetic_algorithm():
    gnome = [1, 2, 3, 4, 0]
    partner = [1, 3, 4, 0, 2]
    print(mate(gnome, partner))

if __name__ == '__main__':
    genetic_algorithm()