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

def get_cloud_centroid(cloud):
    """
    Compute the centroid of the point cloud projected on the X-Z plane.
    """
    _, points = cloud  # Extract points
    x_values = points[:, 0]  # X-axis values
    z_values = points[:, 2]  # Z-axis values
    centroid_x = np.mean(x_values)
    centroid_z = np.mean(z_values)
    return centroid_x, centroid_z

def apply_points_transformation(points, centroid, transformation):
    theta, tx, tz = transformation
    centroid_x, centroid_z = centroid

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
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

    return rotated_points

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

def extract_free_space_polygon(cloud, included_category, excluded_categories):
    free_space_polygon = None
    excluded_polygons = []

    for category_name, points in cloud:
        points_2d = points[:, [0, 2]]  # Project to X-Z plane
        if category_name == included_category:
            free_space_polygon = Polygon(points_2d).convex_hull
        elif category_name in excluded_categories:
            excluded_polygons.append(Polygon(points_2d).convex_hull)

    if not free_space_polygon:
        raise ValueError("Free space not found in the provided point cloud data.")

    if excluded_polygons:
        union_excluded = unary_union(excluded_polygons)
        free_space_polygon = free_space_polygon.difference(union_excluded)

    # Polygon cleaning
    free_space_polygon = clean_polygon(free_space_polygon)      

    return free_space_polygon

def extract_voxels_hashmap(cloud, grid_size):
    # Spatial hash map to store voxels occupied by points in each point cloud
    hash_map = defaultdict(list)
    category_name, points = cloud

    # Calculate cell coordinates for the point
    for point in points:
        voxel_key = tuple((point // grid_size).astype(int)) # X, Y, Z, R, G, B
        hash_map[voxel_key].append(point) # include category_name(mapped into int) for contextual discontinuites

    return hash_map

def maximize_shared_space(loc_polygon, rmt_polygon):

    """
    Calculate intersection areas between polygons in freespaces and boundaries,
    excluding pairs with same index.
    
    Returns sum of all intersection
    """ 
    try:
        intersection = loc_polygon.intersection(rmt_polygon)
        intersection = clean_polygon(intersection)
        intersection_area = intersection.area
    except Exception as e:
        print(f"Warning: Intersection failed for polygons. Using area 0.")
        intersection_area = 0

    return intersection_area

def minimize_discontinuities(keys, loc_voxels, rmt_voxels):
    """
    Minimize discontinuities using A* algorithm, with cost as color differences between voxel pairs.
    
    Args:
        keys (set): Overlapping voxel keys between local and remote voxels.
        loc_voxels (dict): Hashmap of local voxels (key -> list of points).
        rmt_voxels (dict): Hashmap of remote voxels (key -> list of points).
    
    Returns:
        float: Total discontinuity cost for the optimal loop.
        list: List of voxel keys forming the best loop.
    """
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

    # A* Algorithm Loop
    while open_list:
        _, current_key = heappop(open_list)

        # If the loop is formed
        if current_key in closed_list and len(closed_list) >= 10:
            break

        closed_list.add(current_key)

        for neighbor in get_neighbors(current_key):
            if neighbor in closed_list:
                continue

            # Compute color discontinuity cost
            local_points = np.array(loc_voxels.get(neighbor, []))
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

    # Reconstruct the voxel loop path
    best_voxel_loop = []
    node = current_key
    while node in parent_map:
        best_voxel_loop.append(node)
        node = parent_map[node]

    # Ensure the path forms a loop (back to start)
    if len(best_voxel_loop) >= 10 and best_voxel_loop[-1] == start_key:
        return g_cost[current_key], best_voxel_loop
    else:
        return float('inf'), []  # If no valid loop is found

def spea2_optimization(loc_cloud, rmt_cloud, rmt_centroid, in_category, ex_categories, g_size):
    population = initialize_population(pop_size=20, variable_ranges={"theta": [-5, 5], "tx": [-0.3, 0.3], "tz": [-0.3, 0.3]})
    archive = []

    for generation in range(max_generations):
        for individual in population:
            # Apply remote cloud transformation
            remote_transform = (individual["theta"], individual["tx"], individual["tz"])
            transformed_rmt_cloud = apply_points_transformation(rmt_cloud, rmt_centroid, remote_transform)
            
            # Extract free space polygons for obj1
            local_polygon = extract_free_space_polygon(loc_cloud, in_category, ex_categories)
            transformed_remote_polygon = extract_free_space_polygon(transformed_rmt_cloud, in_category, ex_categories)

            # Compute shared space area using polygon
            obj1 = -maximize_shared_space(local_polygon, transformed_remote_polygon)  # Negate for minimization
            
            # Extract voxels hashmap and overlapping hashmap keys for obj2
            local_voxels = extract_voxels_hashmap(loc_cloud, g_size)
            transformed_remote_voxels = extract_voxels_hashmap(transformed_rmt_cloud, g_size)
            overlapping_keys = set(local_voxels.keys()).intersection(set(transformed_remote_voxels.keys()))

            # Compute discontinuities using voxelized hashmap
            obj2, voxel_loop = minimize_discontinuities(overlapping_keys, local_voxels, transformed_remote_voxels)
            
            # Store objectives and voxel loop
            individual["obj1"] = obj1
            individual["obj2"] = obj2
            individual["voxel_loop"] = voxel_loop

        # Combine Population and Archive
        combined_population = population + archive
        
        # # Fitness Assignment and Selection
        # fitness_values = assign_spea2_fitness(combined_population)
        # sorted_population = sort_by_fitness(fitness_values)
        # archive = select_best_solutions(sorted_population, archive_size=10)
        
        # # Generate Offspring (Crossover and Mutation)
        # offspring = generate_offspring(archive, crossover_rate=0.8, mutation_rate=0.2)
        # population = offspring

    return archive  # Pareto-optimal solutions with transformations and voxel loops


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote JSONL file.")
    parser.add_argument("--included_category", type=str, nargs='+', default="ceiling")
    parser.add_argument("--excluded_categories", type=str, nargs='+', default=[
        "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"])
    parser.add_argument("--grid_size", type=float, default=0.5)
    
    args = parser.parse_args()

    # Load point clouds
    local_cloud = jsonl_to_array(args.loc)
    remote_cloud = jsonl_to_array(args.rmt)
    
    # Initialize
    included_category = args.included_category
    excluded_categories = args.excluded_categories
    grid_size = args.grid_size

    # Get centroid of remote cloud
    remote_centroid = get_cloud_centroid(remote_cloud)

    # Run SPEA2 optimization
    pareto_solutions = spea2_optimization(local_cloud, remote_cloud, remote_centroid, included_category, excluded_categories, grid_size)

    # Select the best solution
    best_solution = pareto_solutions[0]
    optimized_transformation = (best_solution["theta"], best_solution["tx"], best_solution["tz"])
    optimized_voxel_loop = best_solution["voxel_loop"]

    # Apply transformation to visualize results
    # transformed_remote = apply_transformation_to_voxels(remote_cloud, optimized_transformation)
    # visualize_transformation(local_cloud, transformed_remote, optimized_voxel_loop)