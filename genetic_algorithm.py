
import numpy as np
import math

from statistics import mean
from optproblems import *

from population import (population, 
                        single_point_crossover,
                        mutate
                        )


class Genetic_Algorithm:
    """
    Runs Genetic Algorithm based on hyper parameters
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
 

    def __init__(self,
                 mutation_rate,
                 crossover_rate,
                 max_iterations,
                 tournament_size,
                 selection_pressure,
                 population_size,
                 num_variables,
                 bounds
                ):
        """
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_iterations = max_iterations
        self.tournament_size = tournament_size
        self.selection_pressure = selection_pressure
        self.population_size = population_size
        self.num_variables = num_variables
        self.bounds = bounds


    def evaluate_population_fitness(self, landscape, population):
        """
        Evaluates fitness of the population based on evaluating every
        genome and finding the best fitness score and best genome in 
        the popuation
        Args:
            landscape - Benchmark Function 
            population - 
        Returns:
            best_genome - best solution 
            best_fitness_score - best fitness score
        """
        best_fitness_score = math.inf
        #check function fitness
        best_genome = None
        #landscape.batch_evaluate(population)
        for genome in population:
            landscape.evaluate(genome)
            local_fitness_score = genome.objective_values

            if local_fitness_score < best_fitness_score:
                best_fitness_score = local_fitness_score
                best_genome = genome

        return best_genome, best_fitness_score


    def tournament_selection(self, population, landscape):
        """
        Divides population into tournaments and Selects best 
        parents from population to mate(crossover and mutate)
        Args:
            population
            landscape
        Returns:
            selected_parents - List of best genomes to mate
        """
        selected_parents = []
        if self.selection_pressure != 1:
            for i in range(len(population)):
                tournament_participants = np.random.choice(population, self.tournament_size)

                if np.random.rand() < self.selection_pressure:
                    tournament_winner, tournament_winner_score = self.evaluate_population_fitness(landscape, tournament_participants)
                    selected_parents.append(tournament_winner)
                else:
                    tournament_winner = np.random.choice(tournament_participants, 1)
                    selected_parents.append(tournament_winner[0])

        else: 
            for i in range(len(population)):
                tournament_participants = np.random.choice(population, self.tournament_size)
                tournament_winner, tournament_winner_score = self.evaluate_population_fitness(landscape, tournament_participants)
                selected_parents.append(tournament_winner)
        return selected_parents


    def evolve(self, population, landscape):
        """
        Calls tournament selection to select parents to mate
        Calls single point cross over for each parent pair
        Calls mutate based on mutation rate and function
        bounds
        Args:
            population
            landscape
        Returns:
            new_generation - New genomes created after GA run
        """
        new_generation = []
        selected_parents = self.tournament_selection(population, landscape)

        for i in range(len(selected_parents)-1):
            cross_genome =  single_point_crossover(selected_parents[i], selected_parents[i+1], self.crossover_rate)
            mutated_genome = mutate(cross_genome, self.mutation_rate, self.bounds)
            new_generation.append(mutated_genome)
        
        return new_generation
        

    def run(self, landscape, function_global_optima):
        """
        Runs the GA 
        - calls population to define initial population
        - Evaluates fitness of initial population 
        to find global optima and global best genome
        - Calls evolve to get new generation 
        - Evaluates fitness of the new generation until 
        number of iterations
        Args:
            landscape
        Returns:
            global_best_genome
            global_best_fitness_score
            best_genomes_history
            best_fitness_score_history
            mean_fitness_score_history            

        """
        best_genomes_history = []
        best_fitness_score_history = []
        mean_fitness_score_history = []
        num_iterations = 0
        init_population = population(self.num_variables, self.population_size,
        self.bounds)

        global_best_genome, global_best_fitness_score = self.evaluate_population_fitness(landscape, init_population)
        best_genomes_history.append(global_best_genome.phenome)
        best_fitness_score_history.append(global_best_fitness_score)

        generation = init_population

        #for i in range(self.max_iterations):
        while global_best_fitness_score - function_global_optima > 1 and num_iterations < self.max_iterations:
        #while function_global_optima - global_best_fitness_score > 1 and num_iterations < self.max_iterations:
            generation = self.evolve(generation, landscape)
            gen_best_genome, gen_best_fitness_score = self.evaluate_population_fitness(landscape, generation)

            if gen_best_fitness_score < global_best_fitness_score:
                global_best_fitness_score = gen_best_fitness_score
                global_best_genome = gen_best_genome
            
            best_genomes_history.append(global_best_genome.phenome)
            best_fitness_score_history.append(global_best_fitness_score)

            #if i % 10 == 0:
            #    mean_fitness_score_history.append(mean(best_fitness_score_history))
            num_iterations += 1
        return global_best_genome, global_best_fitness_score, best_genomes_history, best_fitness_score_history, mean_fitness_score_history, num_iterations