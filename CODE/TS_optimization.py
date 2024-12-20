import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from collections import defaultdict
from heapq import heappop, heappush
from S3DIS_to_json import jsonl_to_array
import random
from tqdm import tqdm

def get_cloud_centroid(cloud):
    """
    Compute the centroid of the point cloud projected on the X-Z plane.
    """
    all_points = np.vstack([points for _, points in cloud])
    x_values = all_points[:, 0]  # X-axis values
    z_values = all_points[:, 2]  # Z-axis values
    centroid_x = np.mean(x_values)
    centroid_z = np.mean(z_values)
    
    return centroid_x, centroid_z

def apply_points_transformation(cloud, centroid, transformation):
    theta, tx, tz = transformation
    centroid_x, centroid_z = centroid

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
    transformed_cloud = []
    for category_name, points in cloud:
        translated_points = np.copy(points)
        translated_points[:, 0] -= centroid_x  # Subtract centroid_x
        translated_points[:, 2] -= centroid_z  # Subtract centroid_z

        # Step 2: Rotate around origin in the X-Z plane
        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])
    
        # Apply rotation to the X-Z components
        rotated_xz = np.dot(translated_points[:, [0, 2]], rotation_matrix.T)
    
        # Update points with rotated X-Z values
        rotated_points = np.copy(translated_points)
        rotated_points[:, 0] = rotated_xz[:, 0]  # Update X
        rotated_points[:, 2] = rotated_xz[:, 1]  # Update Z

        # Step 3: Translate back with offset (tx, tz)
        rotated_points[:, 0] += centroid_x + tx  # Add back centroid and offset tx
        rotated_points[:, 2] += centroid_z + tz  # Add back centroid and offset tz

        transformed_cloud.append((category_name, rotated_points))

    return transformed_cloud

def clean_polygon(polygon):
    """
    Clean and validate a polygon, handling any topological errors
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)
        
    # If the result is a MultiPolygon, take the largest part
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda x: x.area)
            
    return polygon

def extract_free_space_polygon(cloud):
    free_space_polygon = None
    excluded_polygons = []

    for category_name, points in cloud:
        points_2d = points[:, [0, 2]]  # Project to X-Z plane
        if category_name == g_included_category:
            free_space_polygon = Polygon(points_2d).convex_hull
        elif category_name in g_excluded_categories:
            excluded_polygons.append(Polygon(points_2d).convex_hull)

    if not free_space_polygon:
        raise ValueError("Free space not found in the provided point cloud data.")

    if excluded_polygons:
        union_excluded = unary_union(excluded_polygons)
        free_space_polygon = free_space_polygon.difference(union_excluded)

    # Polygon cleaning
    free_space_polygon = clean_polygon(free_space_polygon)

    return free_space_polygon

def extract_voxels_hashmap(cloud):
    # Spatial hash map to store voxels occupied by points in each point cloud
    hash_map = defaultdict(list)
    for category_name, points in cloud:
        # include category_name(mapped into int) for contextual discontinuites
        # Calculate cell coordinates for the point
        for point in points:
            voxel_key = tuple((point // g_grid_size).astype(int))  # X, Y, Z, R, G, B
            hash_map[voxel_key].append(point)
    
    return hash_map

def maximize_shared_space(rmt_polygon, rmt_trans):
    try:
        intersection = g_local_polygon.intersection(rmt_polygon)
        intersection = clean_polygon(intersection)
        intersection_area = intersection.area
        g_shared_polygon[rmt_trans].append(intersection)
    except Exception as e:
        print(f"Warning: Intersection failed for polygons. Using area 0.")
        intersection_area = 0

    return intersection_area

# Nested optimization
def minimize_discontinuities(keys, rmt_trans, rmt_voxels):
    def compute_color_discontinuity(local_points, remote_points):
        """Calculate average color discontinuity (Euclidean distance)."""
        local_avg_color = np.mean(local_points[:, 3:], axis=0)
        remote_avg_color = np.mean(remote_points[:, 3:], axis=0)
        return np.linalg.norm(local_avg_color - remote_avg_color)

    def get_neighbors(current_key):
        """Find neighboring voxel keys (vertical, horizontal, diagonal)."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue  # Skip current voxel
                    neighbor = (current_key[0] + dx, current_key[1] + dy, current_key[2] + dz)
                    if neighbor in keys:  # Check if the neighbor is in valid keys
                        neighbors.append(neighbor)
        return neighbors

    # Initialize A* algorithm
    start_key = next(iter(keys))  # Random start voxel
    open_list = []
    closed_list = set()
    parent_map = {}  # To reconstruct the path later
    g_cost = {key: float('inf') for key in keys}
    f_cost = {key: float('inf') for key in keys}

    # Start node initialization
    g_cost[start_key] = 0
    f_cost[start_key] = 0
    heappush(open_list, (f_cost[start_key], start_key))

    # A* Algorithm Loop with tqdm
    with tqdm(total=len(keys), desc="A* Progress", unit="nodes") as pbar:
        while open_list:
            _, current_key = heappop(open_list)

            # If the loop is formed
            if current_key in closed_list and len(closed_list) >= 10:
                break

            closed_list.add(current_key)
            pbar.update(1)

            for neighbor in get_neighbors(current_key):
                if neighbor in closed_list:
                    continue

                # Compute color discontinuity cost
                local_points = np.array(g_local_voxels.get(neighbor, []))
                remote_points = np.array(rmt_voxels.get(neighbor, []))

                if len(local_points) == 0 or len(remote_points) == 0:
                    continue  # Skip invalid neighbors

                color_cost = compute_color_discontinuity(local_points, remote_points)
                tentative_g_cost = g_cost[current_key] + color_cost

                # Update costs if better path is found
                if tentative_g_cost < g_cost[neighbor]:
                    parent_map[neighbor] = current_key
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = tentative_g_cost  # No heuristic in this case
                    heappush(open_list, (f_cost[neighbor], neighbor))

    # Reconstruct the voxel loop path and save in global voxel loops
    node = current_key
    with tqdm(total=len(parent_map), desc="Reconstructing Path", unit="nodes") as pbar:
        while node in parent_map:
            g_voxel_loops[rmt_trans].append(node)
            node = parent_map[node]
            pbar.update(1)

    # Ensure the path forms a loop (back to start)
    if len(g_voxel_loops[rmt_trans]) >= 10 and g_voxel_loops[rmt_trans][-1] == start_key:
        return g_cost[current_key]  # , best_voxel_loop
    else:
        return float('inf')  # , []  # If no valid loop is found

    

def transitionspace_optimization(theta, tx, tz):
    remote_transformation = (theta, tx, tz)
    transformed_rmt_cloud = apply_points_transformation(g_remote_cloud, g_remote_centroid, remote_transformation)
            
    # Compute shared space area using polygon with transformed remote space
    transformed_remote_polygon = extract_free_space_polygon(transformed_rmt_cloud)
    obj1 = -maximize_shared_space(transformed_remote_polygon, remote_transformation)  # Negate for minimization
    
    # Extract voxels hashmap and overlapping hashmap keys for obj2
    # transformed_remote_voxels = extract_voxels_hashmap(transformed_rmt_cloud)
    # overlapping_keys = set(g_local_voxels.keys()).intersection(set(transformed_remote_voxels.keys()))

    # Compute discontinuities using voxelized hashmap
    # obj2 = minimize_discontinuities(overlapping_keys, remote_transformation, transformed_remote_voxels)
    obj2 = 0
    
    individual = [theta, tx, tz, obj1, obj2]
    return individual

# ======================================= SPEA2 =======================================
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

def euclidean_distance(coordinates):
   a = coordinates
   b = a.reshape(np.prod(a.shape[:-1]), 1, a.shape[-1])
   return np.sqrt(np.einsum('ijk,ijk->ij',  b - a,  b - a)).squeeze()

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

def raw_fitness_function(population, number_of_functions=2):
    """Compute raw fitness values for the population."""
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
    """Calculate fitness using raw fitness and crowding distance."""
    k = int(len(population) ** (1 / 2)) - 1
    fitness = np.zeros((population.shape[0], 1))
    distance = euclidean_distance(population[:, -number_of_functions:])
    for i in range(0, fitness.shape[0]):
        distance_ordered = np.sort(distance[i, :])
        fitness[i, 0] = raw_fitness[i, 0] + 1 / (distance_ordered[k] + 2)
    return fitness

def sort_population_by_fitness(population, fitness):
    """Sort population based on fitness values."""
    idx = np.argsort(fitness[:, 0])
    population = population[idx, :]
    fitness = fitness[idx, :]
    return population, fitness

def breeding(population, fitness, mutation_rate, min_values, max_values):
    """Create offspring population using crossover and mutation."""
    offspring = np.copy(population)
    for i in range(offspring.shape[0]):
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

        # Update offspring
        offspring[i, :3] = [theta, tx, tz]
        result = transitionspace_optimization(theta, tx, tz)
        offspring[i, 3:] = result[3:]

    return offspring

def mutation(population, mutation_rate, min_values, max_values):
    """Mutate the population."""
    for i in range(population.shape[0]):
        if random.random() < mutation_rate:
            theta = population[i, 0] + np.random.uniform(-1, 1)
            tx = population[i, 1] + np.random.uniform(-0.1, 0.1)
            tz = population[i, 2] + np.random.uniform(-0.1, 0.1)

            # Clip to bounds
            theta = np.clip(theta, min_values[0], max_values[0])
            tx = np.clip(tx, min_values[1], max_values[1])
            tz = np.clip(tz, min_values[2], max_values[2])

            # Update individual
            population[i, :3] = [theta, tx, tz]
            result = transitionspace_optimization(theta, tx, tz)
            population[i, 3:] = result[3:]

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
    """Run the SPEA2 optimization algorithm."""
    # Initialize population
    if verbose:
        print("Initializing population...")
    population = np.zeros((population_size, 5))
    with tqdm(total=population_size, desc="Initializing population") as pbar:
        for i in range(population_size):
            theta = np.random.uniform(min_values[0], max_values[0])
            tx = np.random.uniform(min_values[1], max_values[1])
            tz = np.random.uniform(min_values[2], max_values[2])
            individual = transitionspace_optimization(theta, tx, tz)
            population[i, :] = individual
            pbar.update(1)

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
# ======================================= SPEA2 =======================================

def visualize_pareto_front(pereto_set):
    """
    Visualize the best result (pereto_set[0]) with Open3D and matplotlib.
    """
    # Extract the transformation from pereto_set[0]
    theta, tx, tz = pereto_set[:3]
    transformation = (theta, tx, tz)

    # Apply transformation to the remote cloud
    transformed_remote_cloud = apply_points_transformation(g_remote_cloud, g_remote_centroid, transformation)

    # Convert point clouds to Open3D format
    local_all_cloud_points = np.vstack([points for _, points in g_local_cloud])
    remote_all_cloud_points = np.vstack([points for _, points in transformed_remote_cloud])

    print(f"local_all_cloud_points:{local_all_cloud_points[0]}")
    print(f"remote_all_cloud_points:{remote_all_cloud_points[0]}")

    local_cloud_o3d = o3d.geometry.PointCloud()
    local_cloud_o3d.points = o3d.utility.Vector3dVector(local_all_cloud_points[:, :3])

    remote_cloud_o3d = o3d.geometry.PointCloud()
    remote_cloud_o3d.points = o3d.utility.Vector3dVector(remote_all_cloud_points[:, :3])

    # Color the clouds for differentiation
    local_cloud_o3d.colors = o3d.utility.Vector3dVector(local_all_cloud_points[:, 3:6] / 255.0)
    remote_cloud_o3d.colors = o3d.utility.Vector3dVector(remote_all_cloud_points[:, 3:6] / 255.0)

    # Visualize voxel loop as line segments
    voxel_keys = g_voxel_loops[transformation]
    voxel_lines = []
    line_set = o3d.geometry.LineSet()

    for voxel in voxel_keys:
        center = np.array(voxel) * g_grid_size  # Convert voxel keys to real-world positions
        voxel_lines.append(center)

    # Convert to Open3D lines
    line_set.points = o3d.utility.Vector3dVector(voxel_lines)
    line_set.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(voxel_lines) - 1)]
    )
    line_set.paint_uniform_color([0, 1, 0])  # Green for voxel loops

    # Visualize shared polygon
    shared_polygons = g_shared_polygon[transformation]
    shared_meshes = []
    shared_2d_coords = []  # Store 2D (x, z) coordinates for matplotlib visualization

    for polygon in shared_polygons:
        if polygon.is_empty:
            continue
        if polygon.geom_type == "Polygon":
            coords = np.array(polygon.exterior.coords)
            shared_2d_coords.append(coords)
            # Convert 2D to 3D
            coords_3d = np.array([[x, 0.05, z] for x, z in coords])
            shared_mesh = o3d.geometry.LineSet()
            shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
            shared_mesh.lines = o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
            )
            shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
            shared_meshes.append(shared_mesh)
        elif polygon.geom_type == "MultiPolygon":
            for sub_polygon in polygon.geoms:
                coords = np.array(sub_polygon.exterior.coords)
                shared_2d_coords.append(coords)
                # Convert 2D to 3D
                coords_3d = np.array([[x, 0.05, z] for x, z in coords])
                shared_mesh = o3d.geometry.LineSet()
                shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
                shared_mesh.lines = o3d.utility.Vector2iVector(
                    [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
                )
                shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
                shared_meshes.append(shared_mesh)

    # Visualize with matplotlib (2D projection)
    plt.figure(figsize=(10, 10))
    plt.scatter(local_all_cloud_points[:, 0], local_all_cloud_points[:, 2], c=local_all_cloud_points[:, 3:6] / 255.0, s=1, label="Local Cloud")
    plt.scatter(remote_all_cloud_points[:, 0], remote_all_cloud_points[:, 2], c=remote_all_cloud_points[:, 3:6] / 255.0, s=1, label="Transformed Remote Cloud")

    for coords in shared_2d_coords:
        plt.plot(coords[:, 0], coords[:, 1], c='magenta', label="Shared Polygon")

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.title("2D Visualization of Clouds and Shared Polygon")
    plt.show()

    # Visualize all components with Open3D (3D visualization)
    o3d.visualization.draw_geometries(
        [local_cloud_o3d, remote_cloud_o3d, line_set] + shared_meshes
    )




# Global values
g_local_cloud = []
g_local_polygon = []
g_local_voxels = []
g_remote_cloud = []
g_remote_centroid = []
g_grid_size = 0.0
g_included_category = ""
g_excluded_categories = []
g_voxel_loops = defaultdict(list)
g_shared_polygon = defaultdict(list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote JSONL file.")
    parser.add_argument("--grid_size", type=float, default=0.5)
    parser.add_argument("--generation", type=int, default=50)
    
    args = parser.parse_args()

    # Load point clouds
    g_local_cloud = jsonl_to_array(args.loc)  
    g_remote_cloud = jsonl_to_array(args.rmt)

    """
    ValueError: too many values to unpack (expected 2)
    g_local_cloud = 
    [
    ("wall", array([[1.0, 2.0, 3.0, R, G, B], [4.0, 5.0, 6.0], ...])),
    ("ceiling", array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5], ...])),
    ...
    ]
    expected value = ["wall", array([[1.0, 2.0, 3.0, R, G, B], [4.0, 5.0, 6.0], ...])]
    """
    
    # Initialize global values
    g_grid_size = args.grid_size
    g_included_category = "ceiling"
    g_excluded_categories = ["wall", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]
    g_local_polygon = extract_free_space_polygon(g_local_cloud)
    g_local_voxels = extract_voxels_hashmap(g_local_cloud)
    g_remote_centroid = get_cloud_centroid(g_remote_cloud)

    # strength_pareto_evolutionary_algorithm_2 will be placed here***
    pareto_front = strength_pareto_evolutionary_algorithm_2(
                    population_size=20,
                    archive_size=20,
                    mutation_rate=0.1,
                    min_values=[-180.0, -5.0, -5.0],
                    max_values=[180.0, 5.0, 5.0],
                    generations=50,
                    verbose=True,)
    
    # visualize pareto_front[0]
    visualize_pareto_front(pareto_front[0])
    # temp = [60.0, 2.5, 1.0]
    # visualize_pareto_front(temp)
