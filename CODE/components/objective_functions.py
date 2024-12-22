import numpy as np
from heapq import heappop, heappush
import components.geometry_utils as utils
import components.constants as const
from tqdm import tqdm

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
                local_points = np.array(const.g_local_voxels.get(neighbor, []))
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
            const.g_voxel_loops[rmt_trans].append(node)
            node = parent_map[node]
            pbar.update(1)

    # Ensure the path forms a loop (back to start)
    if len(const.g_voxel_loops[rmt_trans]) >= 10 and const.g_voxel_loops[rmt_trans][-1] == start_key:
        return g_cost[current_key]  # , best_voxel_loop
    else:
        return float('inf')  # , []  # If no valid loop is found

def individual_transitionspace(theta, tx, tz):
    remote_transformation = (theta, tx, tz)
    transformed_rmt_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, remote_transformation)
            
    # Compute shared space area using polygon with transformed remote space
    transformed_remote_polygon = utils.extract_free_space_polygon(transformed_rmt_cloud)
    obj1 = -maximize_shared_space(transformed_remote_polygon, remote_transformation)  # Negate for minimization
    
    # # Extract voxels hashmap and overlapping hashmap keys for obj2
    # transformed_remote_voxels = utils.extract_voxels_hashmap(transformed_rmt_cloud)
    # overlapping_keys = set(const.g_local_voxels.keys()).intersection(set(transformed_remote_voxels.keys()))

    # # Compute discontinuities using voxelized hashmap
    # obj2 = minimize_discontinuities(overlapping_keys, remote_transformation, transformed_remote_voxels)
    obj2 = 0
    
    individual = [theta, tx, tz, obj1, obj2]
    return individual