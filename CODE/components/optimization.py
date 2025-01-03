############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: SPEA2 (Strength Pareto Evolutionary Algorithm 2)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: s_ii.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

import os
import numpy as np
import random
from tqdm import tqdm
import components.objective_functions as f

def dominance_function(solution_1, solution_2, number_of_functions=2):
    """
    Determine if solution_1 dominates solution_2 based on objective function values.
    """
    count     = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1[-k] <= solution_2[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance


def euclidean_distance(coordinates):
    """
    Compute pairwise Euclidean distances for a set of coordinates.
    """
    a = coordinates
    b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
    return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()


def roulette_wheel(fitness_new):
    """
    Perform roulette wheel selection based on fitness values.
    """
    fitness = np.zeros((fitness_new.shape[0], 2))
    for i in range(0, fitness.shape[0]):
        fitness[i,0] = 1/(1+ fitness[i,0] + abs(fitness[:,0].min()))
    fit_sum      = fitness[:,0].sum()
    fitness[0,1] = fitness[0,0]
    for i in range(1, fitness.shape[0]):
        fitness[i,1] = (fitness[i,0] + fitness[i-1,1])
    for i in range(0, fitness.shape[0]):
        fitness[i,1] = fitness[i,1]/fit_sum
    ix     = 0
    random = int.from_bytes(os.urandom(8), byteorder = 'big') / ((1 << 64) - 1)
    for i in range(0, fitness.shape[0]):
        if (random <= fitness[i, 1]):
          ix = i
          break
    return ix


def raw_fitness_function(population, number_of_functions=2):
    """
    Compute raw fitness values for the population based on dominance relationships.
    """
    strength = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                if dominance_function(population[i], population[j], number_of_functions):
                    strength[i, 0] += 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if i != j:
                if dominance_function(population[i], population[j], number_of_functions):
                    raw_fitness[j, 0] += strength[i, 0]
    return raw_fitness


def fitness_calculation(population, raw_fitness, number_of_functions=2):
    """
    Calculate fitness using raw fitness and crowding distance.
    """
    k = int(len(population) ** (1 / 2)) - 1
    fitness = np.zeros((population.shape[0], 1))
    distance = euclidean_distance(population[:, -number_of_functions:])
    for i in range(0, fitness.shape[0]):
        distance_ordered = np.sort(distance[i, :])
        fitness[i, 0] = raw_fitness[i, 0] + 1 / (distance_ordered[k] + 2)
    return fitness


def sort_population_by_fitness(population, fitness):
    """
    Sort the population based on fitness values.
    """
    idx = np.argsort(fitness[:, 0])
    population = population[idx, :]
    fitness = fitness[idx, :]
    return population, fitness


def breeding(population, fitness, mutation_rate, min_values, max_values):
    """
    Generate offspring population using crossover and mutation.
    """
    offspring = np.copy(population)
    for i in range(offspring.shape[0]):
        while True:  # Retry until a valid offspring is generated
            parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
            while parent_1 == parent_2:
                parent_2 = roulette_wheel(fitness)

            # Crossover
            theta = (population[parent_1, 0] + population[parent_2, 0]) / 2
            tx = (population[parent_1, 1] + population[parent_2, 1]) / 2
            tz = (population[parent_1, 2] + population[parent_2, 2]) / 2

            # Mutation
            if random.random() < mutation_rate:
                theta += np.random.uniform(-1, 1)
                tx += np.random.uniform(-0.1, 0.1)
                tz += np.random.uniform(-0.1, 0.1)

            # Clip to bounds
            theta = np.clip(theta, min_values[0], max_values[0])
            tx = np.clip(tx, min_values[1], max_values[1])
            tz = np.clip(tz, min_values[2], max_values[2])

            # Generate offspring
            result = f.individual_transitionspace(theta, tx, tz)

            if result is not None:  # Check for valid offspring
                # Update offspring and exit the retry loop
                offspring[i, :3] = [theta, tx, tz]
                offspring[i, 3:] = result[3:]
                break
            else:
                # Optionally log or print a message for debugging
                print(f"Invalid offspring: theta={theta}, tx={tx}, tz={tz}. Retrying...")

    return offspring



def mutation(population, mutation_rate, min_values, max_values):
    """
    Apply mutation to the population.
    """
    for i in range(population.shape[0]):
        if random.random() < mutation_rate:
            while True:  # Retry mutation until a valid result is generated
                # Apply mutation to parameters
                theta = population[i, 0] + np.random.uniform(-1, 1)
                tx = population[i, 1] + np.random.uniform(-0.1, 0.1)
                tz = population[i, 2] + np.random.uniform(-0.1, 0.1)

                # Clip to bounds
                theta = np.clip(theta, min_values[0], max_values[0])
                tx = np.clip(tx, min_values[1], max_values[1])
                tz = np.clip(tz, min_values[2], max_values[2])

                # Generate individual with mutated parameters
                result = f.individual_transitionspace(theta, tx, tz)

                if result is not None:  # Check if the result is valid
                    # Update the individual's parameters in the population
                    population[i, :3] = [theta, tx, tz]
                    population[i, 3:] = result[3:]
                    break  # Exit the retry loop once a valid individual is found
                else:
                    # Optionally log or print for debugging
                    print(f"Invalid mutation: theta={theta}, tx={tx}, tz={tz}. Retrying...")

    return population



def strength_pareto_evolutionary_algorithm_2(
    population_size=10,
    archive_size=10,
    mutation_rate=0.1,
    min_values=[-5, -0.3, -0.3],
    max_values=[5, 0.3, 0.3],
    generations=10,
    verbose=True,
):
    """
    Run the SPEA2 optimization algorithm.
    """
    # Initialize population
    if verbose:
        print("Initializing population...")
    population = np.zeros((population_size, 5))
    with tqdm(total=population_size, desc="Initializing population") as pbar:
        for i in range(population_size):
            while True:  # Keep trying until a valid individual is generated
                theta = np.random.uniform(min_values[0], max_values[0])
                tx = np.random.uniform(min_values[1], max_values[1])
                tz = np.random.uniform(min_values[2], max_values[2])
                individual = f.individual_transitionspace(theta, tx, tz)
                
                if individual is not None:
                    population[i, :] = individual
                    pbar.update(1)
                    break  # Exit the loop once a valid individual is found
                else:
                    # Log or print a message if needed
                    print(f"Invalid individual generated: theta={theta}, tx={tx}, tz={tz}. Retrying...")

    archive = np.zeros((archive_size, 5))

    # Main loop for generations
    for generation in tqdm(range(generations), desc="Generations", leave=True):
        if verbose:
            print(f"Processing Generation {generation}...")

        # Combine population and archive
        combined_population = np.vstack([population, archive])

        # Calculate raw fitness and fitness
        if verbose:
            print("Calculating fitness...")
        with tqdm(total=len(combined_population), desc="Calculating fitness") as pbar:
            raw_fitness = raw_fitness_function(combined_population)
            fitness = fitness_calculation(combined_population, raw_fitness)
            pbar.update(len(combined_population))

        # Sort population by fitness
        if verbose:
            print("Sorting population by fitness...")
        combined_population, fitness = sort_population_by_fitness(
            combined_population, fitness
        )

        # Select top individuals for archive
        archive = combined_population[:archive_size, :]

        # Generate offspring
        if verbose:
            print("Breeding population...")
        with tqdm(total=population_size, desc="Breeding population") as pbar:
            population = breeding(
                combined_population[:population_size, :],
                fitness[:population_size, :],
                mutation_rate,
                min_values,
                max_values,
            )
            pbar.update(population_size)

        # Apply mutation
        if verbose:
            print("Applying mutation...")
        with tqdm(total=population_size, desc="Mutating population") as pbar:
            population = mutation(population, mutation_rate, min_values, max_values)
            pbar.update(population_size)

    return archive
