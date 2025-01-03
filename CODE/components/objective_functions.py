import numpy as np
import open3d as o3d
from collections import defaultdict
from heapq import heappop, heappush
import components.geometry_utils as utils
import components.constants as const
from tqdm import tqdm
import heapq
import random
from copy import deepcopy


def maximize_shared_space(rmt_polygon, rmt_trans):
    try:
        intersection = const.g_local_polygon.intersection(rmt_polygon)
        intersection = utils.clean_polygon(intersection)
        intersection_area = intersection.area
        const.g_shared_polygon[rmt_trans].append(intersection)
    except Exception as e:
        print(f"Warning: Intersection failed for polygons. Using area 0.")
        intersection_area = 0

    return intersection_area

def heuristic(node, goal):
    """Heuristic function: Manhattan distance in 3D."""
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1]) + abs(node[2] - goal[2])

def a_star_3d(graph, start, goal):
    # Priority queue: stores (f_cost, node)
    pq = [(0, start)]
    g_costs = {node: float('inf') for node in graph}
    g_costs[start] = 0
    came_from = {}
    
    while pq:
        _, current = heapq.heappop(pq)
        
        # If the goal is reached, reconstruct the path
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]  # Reverse the path
        
        # Explore neighbors
        for neighbor, weight in graph[current].items():
            g_cost = g_costs[current] + weight
            f_cost = g_cost + heuristic(neighbor, goal)
            
            if g_cost < g_costs[neighbor]:
                g_costs[neighbor] = g_cost
                came_from[neighbor] = current
                heapq.heappush(pq, (f_cost, neighbor))
    
    # If no path is found, raise a RuntimeError
    raise RuntimeError(f"No path found from {start} to {goal}")


def get_neighbors(keys, current_key):
        """Find neighboring voxel keys."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current_key[0] + dx, current_key[1] + dy, current_key[2] + dz)
                    if neighbor in keys:
                        neighbors.append(neighbor)
        return neighbors

def build_graph(keys):
    """Build a weighted graph from voxel keys."""
    graph = defaultdict(dict)
    # Convert to a set of tuples for efficient lookup
    voxel_set = set(map(tuple, keys))
    
    for key in voxel_set:
        neighbors = get_neighbors(voxel_set, key)
        for neighbor in neighbors:
            # Calculate weight using Manhattan distance
            weight = abs(key[0] - neighbor[0]) + abs(key[1] - neighbor[1]) + abs(key[2] - neighbor[2])
            graph[tuple(key)][tuple(neighbor)] = weight  # Ensure keys are tuples
    
    return graph

def initialize_voxel_loop(voxel_keys):
    voxel_keys = np.array(voxel_keys)

    # Step 1: Group by x and find min/max z
    x_groups = defaultdict(list)
    for x, _, z in voxel_keys:
        x_groups[x].append(z)

    prims_xz_min = []  # ordered by x (small -> big)
    prims_xz_max = []  # ordered by x (big -> small)

    for x in sorted(x_groups.keys()):
        z_values = sorted(set(x_groups[x]))
        prims_xz_min.append([x, z_values[0]])
        prims_xz_max.append([x, z_values[-1]])

    prims_xz_max = sorted(prims_xz_max, reverse=True)  # ordered by x (big -> small)
    prims_xz = prims_xz_min + prims_xz_max  # Combine min and max

    # Step 2: Complete prims_xyz with smallest y
    prims_xyz = []
    for x, z in prims_xz:
        # Filter voxel_keys to match x and z, then find smallest y
        matching_voxels = [key for key in voxel_keys if key[0] == x and key[2] == z]
        if matching_voxels:
            smallest_y = min(matching_voxels, key=lambda key: key[1])[1]
            prims_xyz.append((x, smallest_y, z))

    graph = build_graph(voxel_keys)

    # Step 3: Iteratively find the path
    path = []
    for i in range(len(prims_xyz) - 1):
        sub_path = a_star_3d(graph, prims_xyz[i], prims_xyz[i + 1])
        if sub_path:
            # Append sub_path excluding the last node to avoid repetition
            path.extend(sub_path[:-1])

    # Close the loop by connecting back to the start
    final_sub_path = a_star_3d(graph, prims_xyz[-1], prims_xyz[0])
    if final_sub_path:
        path.extend(final_sub_path)

    print("Final path:", path)

    return path

def initialize_voxel_loop2(overlapping_keys, rmt_voxels):
    # Step 1: Group by x and find min/max z
    bound_xz = set()
    for x, _, z in overlapping_keys:
        bound_xz.add((x, z))

    # Step 2: Select all y values if (x, z) is bounded by bound_xz and make extended_keys
    extended_keys = set()
    for x, z in bound_xz:
        for y in range(
            np.min([key[1] for key in const.g_local_voxels.keys() if key[0] == x and key[2] == z]),
            np.max([key[1] for key in const.g_local_voxels.keys() if key[0] == x and key[2] == z]) + 1
        ):
            extended_keys.add((x, y, z))

    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]  # 6-connectivity
    boundary_keys = set()
    
    for key in overlapping_keys:
        for direction in directions:
            neighbor = tuple([key[i] + direction[i] for i in range(3)])
            if neighbor in const.g_local_voxels.keys() and neighbor not in extended_keys:
                boundary_keys.add(key)
                #break  # No need to check other neighbors for this voxel

    return boundary_keys

    

def compute_color_discontinuity(local_points, remote_points):
    """Calculate average color discontinuity (Euclidean distance)."""
    local_avg_color = np.mean(local_points[:, 3:6], axis=0)
    remote_avg_color = np.mean(remote_points[:, 3:6], axis=0)
    return np.linalg.norm(local_avg_color - remote_avg_color)

def compute_geometry_discontinuity(local_points, remote_points):
    """
    Compute the average tangent plane discontinuity using PCA or normal vector cross-product.
    Compare the tangent planes from local and remote points.
    """
    # temporary
    def compute_tangent_pca(points):
        """
        Calculate tangent plane using PCA.
        Returns the normal vector to the plane.
        """
        if len(points) < 3:
            return None  # Not enough points for PCA
        centroid = np.mean(points[:, :3], axis=0)
        centered_points = points[:, :3] - centroid
        covariance_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        return eigenvectors[:, 0]  # Smallest eigenvector as normal vector

    # def compute_tangent_normals(points):
    #     """
    #     Calculate tangent plane normal using cross-product of adjacent normals.
    #     """
    #     if len(points) < 3:
    #         return None  # Not enough points for computation
    #     # Estimate normals (e.g., via Open3D if needed)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    #     pcd.estimate_normals()
    #     normals = np.asarray(pcd.normals)
    #     return np.mean(normals, axis=0)  # Average normal as tangent

    # Use PCA to compute average tangent
    local_tangent_pca = compute_tangent_pca(local_points)
    remote_tangent_pca = compute_tangent_pca(remote_points)

    # # Use normal vector cross-product to compute average tangent
    # local_tangent_normals = compute_tangent_normals(local_points)
    # remote_tangent_normals = compute_tangent_normals(remote_points)

    if local_tangent_pca is None or remote_tangent_pca is None:
        return float('inf')  # No valid tangents, maximum discontinuity

    # Calculate angular difference (cosine similarity) for PCA tangents
    pca_similarity = np.dot(local_tangent_pca, remote_tangent_pca) / (
        np.linalg.norm(local_tangent_pca) * np.linalg.norm(remote_tangent_pca)
    )
    pca_discontinuity = np.arccos(np.clip(pca_similarity, -1.0, 1.0))

    # # Calculate angular difference (cosine similarity) for normal-based tangents
    # normals_similarity = np.dot(local_tangent_normals, remote_tangent_normals) / (
    #     np.linalg.norm(local_tangent_normals) * np.linalg.norm(remote_tangent_normals)
    # )
    # normals_discontinuity = np.arccos(np.clip(normals_similarity, -1.0, 1.0))

    # Return the minimum discontinuity (could also average or prioritize one method)
    # return min(pca_discontinuity, normals_discontinuity)
    return pca_discontinuity

def embed_discontinuities(keys, rmt_voxels):
    """
    Compute and embed color and geometry discontinuities for overlapping voxels.
    """
    embeded_voxels = defaultdict(list)
    # Compute color and geometry discontinuity costs
    for key in keys: 
        local_points = np.array(const.g_local_voxels.get(key, []))
        remote_points = np.array(rmt_voxels.get(key, []))

        assert local_points.size > 0 and remote_points.size > 0, "Both local and remote points should exist for intersection."

        color_cost = compute_color_discontinuity(local_points, remote_points)
        geometry_cost = compute_geometry_discontinuity(local_points, remote_points)
        
        embeded_voxels[key] = (color_cost / 255) + (geometry_cost / 4) # weight here
        # print(f"key: {key}, color_cost: {color_cost}, geometry_cost: {geometry_cost}")
    return embeded_voxels

# Nested optimization
def minimize_discontinuities(keys, rmt_trans, rmt_voxels):
    # Precompute and store discontinuity values in embedded_voxels
    embedded_voxels = embed_discontinuities(keys, rmt_voxels)
    print(f"Embedded voxels count: {len(embedded_voxels)}")

    def get_neighbors(current_key):
        """Find neighboring voxel keys."""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    neighbor = (current_key[0] + dx, current_key[1] + dy, current_key[2] + dz)
                    if neighbor in keys:
                        neighbors.append(neighbor)
        return neighbors

    best_g_cost = float('inf')
    best_loop = None

    # Iterate through all possible start keys
    for start_key in tqdm(keys, desc="Processing start keys"):
        open_list = []
        closed_list = set()
        parent_map = {}
        g_cost = {key: float('inf') for key in keys}
        f_cost = {key: float('inf') for key in keys}

        # Start node initialization
        g_cost[start_key] = 0
        f_cost[start_key] = 0
        heappush(open_list, (f_cost[start_key], start_key))

        while open_list:
            _, current_key = heappop(open_list)

            if current_key in closed_list:
                continue

            closed_list.add(current_key)

            for neighbor in get_neighbors(current_key):
                if neighbor in closed_list:
                    continue

                # Compute cost for neighbor
                neighbor_cost = embedded_voxels[neighbor]
                tentative_g_cost = g_cost[current_key] + neighbor_cost

                # Update costs if better path is found
                if tentative_g_cost < g_cost[neighbor]:
                    parent_map[neighbor] = current_key
                    g_cost[neighbor] = tentative_g_cost
                    f_cost[neighbor] = tentative_g_cost
                    heappush(open_list, (f_cost[neighbor], neighbor))

            # Reconstruct the loop
            loop = []
            node = current_key
            while node in parent_map:
                loop.append(node)
                node = parent_map[node]

            # Close the loop if possible
            if len(loop) >= 10 and loop[-1] == start_key:
                total_discontinuity = sum(embedded_voxels[key] for key in loop)
                if total_discontinuity < best_g_cost:
                    best_g_cost = total_discontinuity
                    best_loop = loop
                    print(f"New best loop found: {best_loop}, Cost: {best_g_cost}")

    # Save the best loop
    if best_loop:
        #const.g_voxel_loops[rmt_trans] = best_loop
        print(f"Best loop saved: {best_loop}")
        return best_g_cost
    else:
        print("No valid loop found.")
        return float('inf')  # Penalty if no loop is formed

def individual_transitionspace(theta, tx, tz):
    remote_transformation = (theta, tx, tz)
    transformed_rmt_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, remote_transformation)
            
    # Compute shared space area using polygon with transformed remote space
    transformed_remote_polygon = utils.extract_free_space_polygon(transformed_rmt_cloud)
    obj1 = -maximize_shared_space(transformed_remote_polygon, remote_transformation)  # Negate for minimization
    # obj1 = 0
    
    # Extract voxels hashmap and overlapping hashmap keys for obj2
    transformed_remote_voxels = utils.extract_voxels_hashmap(transformed_rmt_cloud)
    overlapping_keys = set(const.g_local_voxels.keys()).intersection(set(transformed_remote_voxels.keys()))

    # for i in overlapping_keys:
    #     print(i)
    # Compute discontinuities and embed the value
    #embeded_voxels = embed_discontinuities(overlapping_keys, transformed_remote_voxels)

    # Initialize voxel loop
    if len(overlapping_keys) == 0:
        return None
    const.g_voxel_loops[remote_transformation] = initialize_voxel_loop2(overlapping_keys, transformed_remote_voxels)
    #overlapping_keys_list = list(overlapping_keys)
    #const.g_voxel_loops[remote_transformation] = initialize_voxel_loop(overlapping_keys_list)

    # Compute total discontinuities using voxelized hashmap
    #obj2 = minimize_discontinuities(overlapping_keys, remote_transformation, transformed_remote_voxels)
    
    # obj1 test
    obj2 = 0
    
    individual = [theta, tx, tz, obj1, obj2]
    return individual
