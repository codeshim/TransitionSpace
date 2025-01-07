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


def calculate_shared_space(rmt_polygon, rmt_trans):
    try:
        intersection = const.g_local_polygon.intersection(rmt_polygon)
        intersection = utils.clean_polygon(intersection)
        intersection_area = intersection.area
    except Exception as e:
        print(f"Warning: Intersection failed for polygons. Using area 0.")
        intersection_area = 0

    return intersection_area, intersection

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


def individual_transitionspace(theta, tx, tz):
    remote_transformation = (theta, tx, tz)
    transformed_rmt_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, remote_transformation)
            
    # # Compute shared space area using polygon with transformed remote space
    # transformed_remote_polygon = utils.extract_free_space_polygon(transformed_rmt_cloud)
    # area, shared_space = calculate_shared_space(transformed_remote_polygon, remote_transformation)  # Negate for minimization
    # obj1 = -area

    # Extract voxels hashmap of transformed remote space
    trans_rmt_strt_voxels = utils.extract_selected_voxels_keys(
        transformed_rmt_cloud, const.g_structure_categories)
    trans_rmt_feat_voxels = utils.extract_selected_voxels_keys(
        transformed_rmt_cloud, const.g_feature_categories)
    
    # maximize overlapped structures
    if const.g_loc_strt_voxels.ndim == 2 and trans_rmt_strt_voxels.ndim == 2:
        # View arrays as 1D structured arrays for row-wise comparison
        loc_voxels = const.g_loc_strt_voxels.view([('', const.g_loc_strt_voxels.dtype)] * const.g_loc_strt_voxels.shape[1])
        trans_voxels = trans_rmt_strt_voxels.view([('', trans_rmt_strt_voxels.dtype)] * trans_rmt_strt_voxels.shape[1])

        # Find the intersection of rows
        overlapping_strt_voxels = np.intersect1d(loc_voxels, trans_voxels)

        # Convert back to 2D array if needed
        overlapping_strt_voxels = overlapping_strt_voxels.view(const.g_loc_strt_voxels.dtype).reshape(-1, const.g_loc_strt_voxels.shape[1])
    else:
        raise ValueError("Voxel arrays must be 2D for row-wise comparison.")
    
    obj1 = -len(overlapping_strt_voxels)

    if obj1 == 0:
        return None

    # minimize overlapped features
    if const.g_loc_feat_voxels.ndim == 2 and trans_rmt_feat_voxels.ndim == 2:
        # View arrays as 1D structured arrays for row-wise comparison
        loc_voxels = const.g_loc_feat_voxels.view([('', const.g_loc_feat_voxels.dtype)] * const.g_loc_feat_voxels.shape[1])
        trans_voxels = trans_rmt_feat_voxels.view([('', trans_rmt_feat_voxels.dtype)] * trans_rmt_feat_voxels.shape[1])

        # Find the intersection of rows
        overlapping_feat_voxels = np.intersect1d(loc_voxels, trans_voxels)

        # Convert back to 2D array if needed
        overlapping_feat_voxels = overlapping_feat_voxels.view(const.g_loc_feat_voxels.dtype).reshape(-1, const.g_loc_feat_voxels.shape[1])
    else:
        raise ValueError("Voxel arrays must be 2D for row-wise comparison.")
    obj2 = len(overlapping_feat_voxels)

    # Ensure obj1 and obj2 fall within bounds
    if const.g_obj1_min <= -obj1 <= const.g_obj1_max and const.g_obj2_min <= obj2 <= const.g_obj2_max:
        # Normalize obj1 (negated for minimization)
        normalized_obj1 = (-obj1 - const.g_obj1_min) / (const.g_obj1_max - const.g_obj1_min)

        # Normalize obj2
        normalized_obj2 = (obj2 - const.g_obj2_min) / (const.g_obj2_max - const.g_obj2_min)
    else:
        raise ValueError("obj1 or obj2 is out of bounds.")
    
    obj1 = normalized_obj1
    obj2 = normalized_obj2

    # Save the overlapped voxels
    const.g_overlap_strt_voxels[remote_transformation] = overlapping_strt_voxels
    const.g_overlap_feat_voxels[remote_transformation] = overlapping_feat_voxels

    individual = [theta, tx, tz, obj1, obj2]
    #print(f"individual: theta({theta}), tx({tx}), tz({tz}), obj1({obj1}), obj2({obj2})")
    return individual
