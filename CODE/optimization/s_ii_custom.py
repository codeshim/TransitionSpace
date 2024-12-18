############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Metaheuristics
# Lesson: SPEA2 (Strength Pareto Evolutionary Algorithm 2)

# Citation: 
# PEREIRA, V. (2021). Project: pyMultiojective, File: s_ii.py, GitHub repository: <https://github.com/Valdecy/pyMultiojective>

############################################################################

# Required Libraries
import numpy  as np
import random
import os
from shapely import affinity
from functools import reduce

import matplotlib
import matplotlib.pyplot as plt
from spea2 import plot_polygon, visualize_optimization_step

############################################################################

# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

def get_relative_position(child, parent):
    """
    Calculate child's position relative to parent
    Returns vector from parent's centroid to child's centroid
    """
    parent_centroid = parent.centroid
    child_centroid = child.centroid
    return (child_centroid.x - parent_centroid.x, 
            child_centroid.y - parent_centroid.y)

def encode_polygon_state(polygon):
    """
    Encode polygon position as [x_translation, y_translation, rotation_angle]
    """
    return np.array([0, 0, 0])  # Initial state with no transformation


############################################################################


def initial_population_from_polygons(input_polygons, boundaries, list_of_functions):
    """
    Create initial population encoding translation and rotation parameters
    """
    population_size = len(input_polygons)
    # For each polygon: [x_trans, y_trans, rotation]
    state_size = 3
    population = np.zeros((population_size, state_size * len(input_polygons) + len(list_of_functions)))
    
    for i in range(population_size):
        # Initialize random transformations for each polygon
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            # Random translation within bounds 
            population[i, base_idx:base_idx+2] = np.random.uniform(-0.3, 0.3, 2)
            # Random rotation 
            population[i, base_idx+2] = np.random.uniform(-5, 5)
            
        # Calculate objective functions
        transformed_polygons = []
        transformed_boundaries = []
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            state = population[i, base_idx:base_idx+3]
            trans_poly, trans_bound = transform_polygon_with_child(
                input_polygons[j], 
                boundaries[j], 
                state
            )
            transformed_polygons.append(trans_poly)
            transformed_boundaries.append(trans_bound)
            
        for k, func in enumerate(list_of_functions):
            if func.__name__ == 'maximize_aligned_area':
                population[i, -(k+1)] = func(transformed_polygons, transformed_boundaries)
            else:  # minimize_overlapped_area
                population[i, -(k+1)] = func(transformed_boundaries)
            
    return population

# Function: Dominance
def dominance_function(solution_1, solution_2, number_of_functions = 2):
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

# Function: Raw Fitness
def raw_fitness_function(population, number_of_functions = 2):
    strength    = np.zeros((population.shape[0], 1))
    raw_fitness = np.zeros((population.shape[0], 1))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if (i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    strength[i,0] = strength[i,0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if (i != j):
                if dominance_function(solution_1 = population[i,:], solution_2 = population[j,:], number_of_functions = number_of_functions):
                    raw_fitness[j,0] = raw_fitness[j,0] + strength[i,0]
    return raw_fitness

# Function: Build Distance Matrix
def euclidean_distance(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

# Function: Fitness
def fitness_calculation(population, raw_fitness, number_of_functions = 2):
    k        = int(len(population)**(1/2)) - 1
    fitness  = np.zeros((population.shape[0], 1))
    distance = euclidean_distance(population[:,population.shape[1]-number_of_functions:])
    for i in range(0, fitness.shape[0]):
        distance_ordered = (distance[distance[:,i].argsort()]).T
        fitness[i,0]     = raw_fitness[i,0] + 1/(distance_ordered[i,k] + 2)
    return fitness

# Function: Sort Population by Fitness
def sort_population_by_fitness(population, fitness):
    idx        = np.argsort(fitness[:,-1]).tolist()
    fitness    = fitness[idx,:]
    population = population[idx,:]
    return population, fitness

# Function: Selection
def roulette_wheel(fitness_new): 
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

def breeding(population, fitness, input_polygons, boundaries, mu=1, list_of_functions=[]):
    offspring = np.copy(population)
    state_size = 3  # [x_trans, y_trans, rotation]
    
    for i in range(offspring.shape[0]):
        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
        while parent_1 == parent_2:
            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
            
        # Crossover for each polygon's parameters
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            for k in range(state_size):
                idx = base_idx + k
                rand = random.random()
                if rand < 0.5:
                    offspring[i, idx] = (population[parent_1, idx] + population[parent_2, idx]) / 2
                    
        # Calculate objective functions
        transformed_polygons = []
        transformed_boundaries = []
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            state = offspring[i, base_idx:base_idx+3]
            trans_poly, trans_bound = transform_polygon_with_child(
                input_polygons[j], 
                boundaries[j], 
                state
            )
            transformed_polygons.append(trans_poly)
            transformed_boundaries.append(trans_bound)
            
        for k, func in enumerate(list_of_functions):
            if func.__name__ == 'maximize_aligned_area':
                offspring[i, -(k+1)] = func(transformed_polygons, transformed_boundaries)
            else:  # minimize_overlapped_area
                offspring[i, -(k+1)] = func(transformed_boundaries)
            
    return offspring



def mutation(offspring, mutation_rate, input_polygons, boundaries, eta=1, list_of_functions=[]):
    state_size = 3  # [x_trans, y_trans, rotation]
    
    for i in range(offspring.shape[0]):
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            for k in range(state_size):
                if random.random() < mutation_rate:
                    if k < 2:  # Translation
                        offspring[i, base_idx + k] += random.gauss(0, 2)
                    else:  # Rotation
                        offspring[i, base_idx + k] += random.gauss(0, 1)
                        offspring[i, base_idx + k] = np.clip(offspring[i, base_idx + k], -5, 5)
                        
        # Recalculate objective functions
        transformed_polygons = []
        transformed_boundaries = []
        for j in range(len(input_polygons)):
            base_idx = j * state_size
            state = offspring[i, base_idx:base_idx+3]
            #print(state)
            trans_poly, trans_bound = transform_polygon_with_child(
                input_polygons[j], 
                boundaries[j], 
                state
            )
            transformed_polygons.append(trans_poly)
            transformed_boundaries.append(trans_bound)
            
        for k, func in enumerate(list_of_functions):
            if func.__name__ == 'maximize_aligned_area':
                offspring[i, -(k+1)] = func(transformed_polygons, transformed_boundaries)
            else:  # minimize_overlapped_area
                offspring[i, -(k+1)] = func(transformed_boundaries)
            
    return offspring



############################################################################
# logger 
def verify_relative_position(polygon, boundary, name=""):
    """
    Verify and print the relative position between parent and child
    """
    parent_centroid = polygon.centroid
    child_centroid = boundary.centroid
    relative_x = child_centroid.x - parent_centroid.x
    relative_y = child_centroid.y - parent_centroid.y
    print(f"{name} Parent centroid: ({parent_centroid.x:.2f}, {parent_centroid.y:.2f})")
    print(f"{name} Child centroid: ({child_centroid.x:.2f}, {child_centroid.y:.2f})")
    print(f"{name} Relative position: ({relative_x:.2f}, {relative_y:.2f})")
    print("-" * 50)
    return relative_x, relative_y


def transform_polygon_with_child(polygon, boundary, state, debug=False):
    """
    Transform both parent polygon and its child boundary while maintaining their relative position
    
    Args:
        polygon: Parent polygon (shapely Polygon)
        boundary: Child polygon (shapely Polygon)
        state: Tuple of (x_translation, y_translation, rotation_degrees)
        debug: Boolean to enable debug printing
        
    Returns:
        Tuple of (transformed_parent, transformed_child)
    """
    x_trans, y_trans, rotation = state
    
    if debug:
        print(f"\nTransformation parameters: dx={x_trans:.2f}, dy={y_trans:.2f}, rotation={rotation:.2f}")
        print("Before transformation:")
        initial_rel_x, initial_rel_y = verify_relative_position(polygon, boundary, "Before ")
    
    # Get original centroids
    parent_centroid = polygon.centroid
    child_centroid = boundary.centroid
    
    # Calculate initial relative position of child with respect to parent
    initial_relative_x = child_centroid.x - parent_centroid.x
    initial_relative_y = child_centroid.y - parent_centroid.y
    
    # Step 1: Rotate both polygons around parent's centroid
    parent_rotated = affinity.rotate(polygon, rotation, origin=parent_centroid)
    child_rotated = affinity.rotate(boundary, rotation, origin=parent_centroid)
    
    # Step 2: Apply translation to both polygons
    parent_transformed = affinity.translate(parent_rotated, x_trans, y_trans)
    child_transformed = affinity.translate(child_rotated, x_trans, y_trans)
    
    if debug:
        print("\nAfter transformation:")
        final_rel_x, final_rel_y = verify_relative_position(parent_transformed, child_transformed, "After ")
        
        # Calculate the expected relative position after rotation
        angle_rad = np.radians(rotation)
        expected_rel_x = (initial_rel_x * np.cos(angle_rad) - 
                         initial_rel_y * np.sin(angle_rad))
        expected_rel_y = (initial_rel_x * np.sin(angle_rad) + 
                         initial_rel_y * np.cos(angle_rad))
        
        # Verify relative position is maintained
        rel_pos_maintained = (abs(final_rel_x - expected_rel_x) < 1e-10 and 
                            abs(final_rel_y - expected_rel_y) < 1e-10)
        print(f"\nExpected relative position: dx={expected_rel_x:.6f}, dy={expected_rel_y:.6f}")
        print(f"Relative position maintained: {rel_pos_maintained}")
    
    return parent_transformed, child_transformed



############################################################################

# SPEA-2 Function
def strength_pareto_evolutionary_algorithm_2(population_size=5, archive_size=5, mutation_rate=0.1, 
                                          input_polygons=[], boundaries=[], 
                                          list_of_functions=[], generations=50, mu=1, eta=1, verbose=True):
    count = 0
    population = initial_population_from_polygons(input_polygons, boundaries, list_of_functions)
    archive = initial_population_from_polygons(input_polygons, boundaries, list_of_functions)
    

    # logger: initial positions
    print("\ninitial positions:")
    for i in range(len(input_polygons)):
        verify_relative_position(input_polygons[i], boundaries[i], f"Polygon {i} Initial ")


    while count <= generations:
        if verbose:
            print('Generation = ', count)

            # logger: Verify movement during optimization (for first polygon in population)
            if count % 10 == 0:  # Check every 10 generations to avoid too much output
                print(f"\nChecking movement at generation {count}:")
                state = population[0, :3]  # Get transformation state for first polygon
                trans_poly, trans_bound = transform_polygon_with_child(
                    input_polygons[0], 
                    boundaries[0], 
                    state,
                    debug=True
                )
        
        population = np.vstack([population, archive])
        raw_fitness = raw_fitness_function(population, number_of_functions=len(list_of_functions))
        fitness = fitness_calculation(population, raw_fitness, number_of_functions=len(list_of_functions))
        
        population, fitness = sort_population_by_fitness(population, fitness)
        population, archive, fitness = (population[0:population_size,:], 
                                      population[0:archive_size,:], 
                                      fitness[0:archive_size,:])
        
        population = breeding(population, fitness, input_polygons, boundaries, 
                            mu=mu, list_of_functions=list_of_functions)
        population = mutation(population, mutation_rate, input_polygons, boundaries, 
                            eta=eta, list_of_functions=list_of_functions)
        

        # visualizer
        # Visualize current state (you might want to do this every N generations to avoid too many images)
        if count % 10 == 0:  # Save every 10th generation
            # Get transformed polygons from the best solution in archive
            best_solution = archive[0]  # Assuming archive is sorted by fitness
            transformed_polygons = []
            transformed_boundaries = []
            
            for i in range(len(input_polygons)):
                state = best_solution[i*3:(i*3)+3]
                trans_poly, trans_bound = transform_polygon_with_child(
                    input_polygons[i],
                    boundaries[i],
                    state
                )
                transformed_polygons.append(trans_poly)
                transformed_boundaries.append(trans_bound)
            
            visualize_optimization_step(
                input_polygons,
                boundaries,
                transformed_polygons,
                transformed_boundaries,
                "./images",
                frame=count+100
            )



        count += 1
    
    '''
    # logger: Verify final positions
    print("\nVerifying final positions:")
    best_solution = archive[0]
    for i in range(len(input_polygons)):
        state = best_solution[i*3:(i*3)+3]
        trans_poly, trans_bound = transform_polygon_with_child(
            input_polygons[i],
            boundaries[i],
            state,
            debug=True
        )
    '''

        
    return archive
