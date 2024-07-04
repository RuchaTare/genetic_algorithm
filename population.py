import numpy as np
from optproblems import *


def population(num_variables=None, population_size=None, 
    bounds=None):
    """ 
    Creates initial population randomly between bound values
    based on dimensions in the population size
    and number of variables in each dimension
    Args:
        num_variables - Dimensions of function
        population_size - Size of array
    Return:
        population - num of variables * population size
    """
    num_variables = num_variables
    population_size = population_size
    if population_size is not None and num_variables is not None:
        population = [Individual(np.random.uniform(bounds[0], bounds[1], 
                                                        num_variables)) for _ in range(population_size)]
    return population

def single_point_crossover(parent1, parent2, crossover_rate=0.5):
    """
    Crossover creates child from parent genomes by crossing over 
    randomly at a cross over point 
    if crossover rate is more than 0.7 then parent genome is directly 
    copied to the new child created
    Args:
        parent1 - Genome 1
        parent2 - Genome 2
        crossover_rate - Probability of crossing over
    Return:
        Child - New genome created
    """

    if crossover_rate <= 0.7:
        crossover_point = np.random.randint(1, len(parent1.phenome))
        child = Individual(np.concatenate((parent1.phenome[crossover_point:], parent2.phenome[:crossover_point])))
    else:
        child = np.random.choice([parent1, parent2])

    return child


def mutate(child, mutation_rate=0.5, boundary_limits=[-100, 100]):
    """ 
    Every child has probability to mutate randomly between the function
    limits (-100, 100)
    mutate replaces current child genome value with random value over
    uniform distribution
    Args:
        child - Child value after cross over
        mutation_rate - probability of mutation 
        boundry_limits - limits of the function
    Return:
        child - Mutated child value
    """
    child_size = child.phenome.shape[0]

    # generate random mutation_probability for each gene in the child (0, 1)
    mutation_probability = np.random.uniform(0, 1, child_size)

    # based on mutation_probability vs mutation_rate, gene will be altered by mutation
    for i in range(child_size):
        if mutation_probability[i] < mutation_rate:
            child.phenome[i] = np.random.uniform(boundary_limits[0], boundary_limits[1], 1)
        
    return child
