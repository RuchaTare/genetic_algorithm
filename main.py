import numpy as np
from optproblems.cec2005 import F2, F1, F6, F15

from genetic_algorithm import Genetic_Algorithm


def main():
    """
    Main method calls Genetic Algorithm to find global minimum value
    of benchmark functions used 
    Args:
        mutation_rate - Probability of mutation of genome
        crossover_rate - Probability of parents pair cross over to form child
        max_iterations - number of iterations
        tournament_size - Tournament size to divide population to find best parents
        selection_pressure - probability of getting best fit parents
        population_size - Population array size
        num_variables - dimensions of function
        bounds
    Return:
        global_best_genome - best solution 
        global_best_fitness_score - best fitness score from GA
        best_genomes_history - All the list of genome solutions  
        best_fitness_score_history - List of best fitness score
        mean_fitness_score_history - mean fitness over all iterations
    """
    mutation_rate = 0.1
    crossover_rate = 0.7
    max_iterations = 500
    tournament_size = 50
    selection_pressure = 0.9
    population_size = 100
    num_variables = 10
    bounds = [-100, 100]

    ga = Genetic_Algorithm(mutation_rate,
                           crossover_rate,
                           max_iterations,
                           tournament_size,
                           selection_pressure,
                           population_size,
                           num_variables,
                           bounds)

    landscape = F1(num_variables)
    global_optima = landscape.get_optimal_solutions()
    landscape.batch_evaluate(global_optima)
    for x in global_optima:
        function_global_optima = x.objective_values


    global_best_genome, global_best_fitness_score, best_genomes_history, best_fitness_score_history, mean_fitness_score_history, num_iterations = ga.run(landscape, function_global_optima)

    print(f"global_best_genome: {global_best_genome.phenome}")
    print(f"global_best_fitness_score: {global_best_fitness_score}")
    print(f"Function global optima: {function_global_optima}")
    # print(f"best_genomes_history: {best_genomes_history}")
    print(f"best_fitness_score_history: {best_fitness_score_history}")
    # print(f"mean_fitness_score_history: {mean_fitness_score_history}")
    print(f"number of iterations: {num_iterations}")


    return global_best_genome, global_best_fitness_score, best_genomes_history, best_fitness_score_history, mean_fitness_score_history


if __name__ == "__main__":
    main()
