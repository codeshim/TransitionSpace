# ------------ objective_function.py ------------ 
import numpy as np
import open3d as o3d
from collections import defaultdict
from heapq import heappop, heappush
import components.geometry_utils as utils
import components.constants as const
from tqdm import tqdm
import random
from copy import deepcopy


def calculate_shared_space(rmt_polygon):
    intersection = None
    try:
        intersection = const.g_local_polygon.intersection(rmt_polygon)
        intersection = utils.clean_polygon(intersection)
        intersection_area = intersection.area
    except Exception as e:
        print(f"Warning: Intersection failed for polygons. Using area 0.")
        intersection_area = 0

    return intersection_area, intersection


def individual_transitionspace(theta, tx, tz):
    remote_transformation = (theta, tx, tz)

    if (not const.g_isallvoxel):
        # ============================ Maximize overlapped structure polygons ============================
        transformed_rmt_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, remote_transformation)
        # Compute shared space area using polygon with transformed remote space
        transformed_remote_polygon = utils.extract_free_space_polygon(transformed_rmt_cloud)
        area, shared_space = calculate_shared_space(transformed_remote_polygon, remote_transformation)  # Negate for minimization
        obj1 = -area
        # ============================ Maximize overlapped structure polygons ============================
    else:
        # ============================= Maximize overlapped structure voxels =============================
        transformed_rmt_strt_points = utils.apply_transformation_local_points(const.g_remote_strt_points, 
                                                                              const.g_remote_centroid, 
                                                                              remote_transformation)
        transfromed_rmt_strt_voxels = utils.extract_voxels_keys_points(transformed_rmt_strt_points)
        overlapping_strt_voxels = utils.extract_intersected_voxels(const.g_loc_strt_voxels, transfromed_rmt_strt_voxels)
        obj1 = -len(overlapping_strt_voxels)
        # ============================= Maximize overlapped structure voxels =============================

    if obj1 == 0:
        return None
    
    if (const.g_ismulitobj):
        # ======================================== Multi-objectives ========================================
        transformed_rmt_feat_points = utils.apply_transformation_local_points(const.g_remote_feat_points, 
                                                                              const.g_remote_centroid, 
                                                                              remote_transformation)
        transfromed_rmt_feat_voxels = utils.extract_voxels_keys_points(transformed_rmt_feat_points)
        overlapping_feat_voxels = utils.extract_intersected_voxels(const.g_loc_feat_voxels, transfromed_rmt_feat_voxels)
        obj2 = len(overlapping_feat_voxels)
        # ======================================== Multi-objectives ========================================
    
    # Normalize obj1 (negated for minimization)
    normalized_obj1 = (obj1 - const.g_obj1_min) / (const.g_obj1_max - const.g_obj1_min)

    # Normalize obj2
    if (const.g_ismulitobj):
        normalized_obj2 = (obj2 - const.g_obj2_min) / (const.g_obj2_max - const.g_obj2_min)
    else: normalized_obj2 = 0.0
    
    obj1 = normalized_obj1
    obj2 = normalized_obj2

    if (const.g_isallvoxel):
        const.g_overlap_strt_voxels[remote_transformation] = overlapping_strt_voxels
    if (const.g_ismulitobj):
        const.g_overlap_feat_voxels[remote_transformation] = overlapping_feat_voxels

    individual = [theta, tx, tz, obj1, obj2]
    #print(f"individual: theta({theta}), tx({tx}), tz({tz}), obj1({obj1}), obj2({obj2})")
    return individual

def obj2_transitionspace(theta, tx, tz):
    remote_transformation = (theta, tx, tz)
    transformed_rmt_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, remote_transformation)
    
    # Extract voxels hashmap of transformed remote space
    trans_rmt_feat_voxels = utils.extract_selected_voxels_keys(
        transformed_rmt_cloud, const.g_feature_categories)

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
    if const.g_obj2_min <= obj2 <= const.g_obj2_max:
        # Normalize obj2
        normalized_obj2 = (obj2 - const.g_obj2_min) / (const.g_obj2_max - const.g_obj2_min)

    obj2 = normalized_obj2

    const.g_overlap_feat_voxels[remote_transformation] = overlapping_feat_voxels

    return obj2

