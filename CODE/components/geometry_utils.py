# ------------ geometry_utils.py ------------ 
import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from collections import defaultdict
import components.constants as const

def downsample_points(group_clouds, voxel_size=0.05):
    down_group_clouds = []
    total_original_points = 0
    total_downsampled_points = 0
    
    print("\nDownsampling Statistics:")
    print("-" * 50)
    
    for category_name, points in group_clouds:
        try:
            original_count = len(points)
            total_original_points += original_count
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            xyz = points[:, :3].astype(np.float64)
            rgb = points[:, 3:6].astype(np.float64) / 255.0
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            
            # Downsample
            down_pcd = pcd.voxel_down_sample(voxel_size)
            
            # Convert back
            down_points = np.asarray(down_pcd.points)
            down_colors = (np.asarray(down_pcd.colors) * 255).astype(np.uint8)
            downsampled_data = np.hstack((down_points, down_colors))
            
            downsampled_count = len(down_points)
            total_downsampled_points += downsampled_count
            
            reduction = original_count - downsampled_count
            
            down_group_clouds.append((category_name, downsampled_data))
            
            del pcd
            del down_pcd
            
        except Exception as e:
            print(f"Error processing category {category_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    # Print total statistics
    total_reduction = total_original_points - total_downsampled_points
    total_reduction_percentage = (total_reduction / total_original_points) * 100
    print("\nTotal Statistics:")
    print(f"Total original points: {total_original_points:,}")
    print(f"Total downsampled points: {total_downsampled_points:,}")
    print(f"Total reduction: {total_reduction:,} points ({total_reduction_percentage:.1f}%)")
    
    return down_group_clouds


def get_cloud_centroid(group_clouds):
    """
    Compute the centroid of the point cloud projected on the X-Z plane.
    """
    all_points = np.vstack([points for _, points in group_clouds])
    x_values = all_points[:, 0]  # X-axis values
    z_values = all_points[:, 2]  # Z-axis values
    centroid_x = np.mean(x_values)
    centroid_z = np.mean(z_values)

    return centroid_x, centroid_z


def apply_points_transformation(group_clouds, centroid, transformation):
    """
    Apply a transformation to a point cloud based on rotation and translation.
    """
    theta, tx, tz = transformation
    centroid_x, centroid_z = centroid

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
    transformed_cloud = []
    for category_name, points in group_clouds:
        translated_points = np.copy(points)
        translated_points[:, 0] -= centroid_x
        translated_points[:, 2] -= centroid_z

        # Step 2: Rotate around origin in the X-Z plane
        rotation_matrix = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ])
        rotated_xz = np.dot(translated_points[:, [0, 2]], rotation_matrix.T)

        rotated_points = np.copy(translated_points)
        rotated_points[:, 0] = rotated_xz[:, 0]
        rotated_points[:, 2] = rotated_xz[:, 1]

        # Step 3: Translate back with offset (tx, tz)
        rotated_points[:, 0] += centroid_x + tx
        rotated_points[:, 2] += centroid_z + tz

        transformed_cloud.append((category_name, rotated_points))

    return transformed_cloud

def apply_transformation_local_points(points, centroid, transformation):
    theta, tx, tz = transformation
    centroid_x, centroid_z = centroid

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
    translated_points = np.copy(points)
    translated_points[:, 0] -= centroid_x
    translated_points[:, 2] -= centroid_z

    # Step 2: Rotate around origin in the X-Z plane
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    rotated_xz = np.dot(translated_points[:, [0, 2]], rotation_matrix.T)

    rotated_points = np.copy(translated_points)
    rotated_points[:, 0] = rotated_xz[:, 0]
    rotated_points[:, 2] = rotated_xz[:, 1]

    # Step 3: Translate back with offset (tx, tz)
    rotated_points[:, 0] += centroid_x + tx
    rotated_points[:, 2] += centroid_z + tz

    return rotated_points

def apply_transformation_points(points, transformation):
    """
    Apply a transformation to a point cloud based on rotation and translation.
    """
    theta, tx, tz = transformation
    x_values = points[:, 0]  # X-axis values
    z_values = points[:, 2]  # Z-axis values
    centroid_x = np.mean(x_values)
    centroid_z = np.mean(z_values)

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
    translated_points = np.copy(points)
    translated_points[:, 0] -= centroid_x
    translated_points[:, 2] -= centroid_z

    # Step 2: Rotate around origin in the X-Z plane
    rotation_matrix = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)]
    ])
    rotated_xz = np.dot(translated_points[:, [0, 2]], rotation_matrix.T)

    rotated_points = np.copy(translated_points)
    rotated_points[:, 0] = rotated_xz[:, 0]
    rotated_points[:, 2] = rotated_xz[:, 1]

    # Step 3: Translate back with offset (tx, tz)
    rotated_points[:, 0] += centroid_x + tx
    rotated_points[:, 2] += centroid_z + tz

    return rotated_points


def clean_polygon(polygon):
    """
    Clean and validate a polygon, handling any topological errors.
    """
    if not polygon.is_valid:
        polygon = make_valid(polygon)

    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda x: x.area)

    return polygon


def extract_free_space_polygon(group_clouds):
    """
    Extract a polygon representing the free space in the point cloud.
    """
    free_space_polygon = None
    excluded_polygons = []

    for category_name, points in group_clouds:
        points_2d = points[:, [0, 2]]
        if category_name == const.g_included_category:
            free_space_polygon = Polygon(points_2d).convex_hull
        elif category_name in const.g_excluded_categories:
            excluded_polygons.append(Polygon(points_2d).convex_hull)

    if not free_space_polygon:
        raise ValueError("Free space not found in the provided point cloud data.")

    if excluded_polygons:
        union_excluded = unary_union(excluded_polygons)
        free_space_polygon = free_space_polygon.difference(union_excluded)

    return clean_polygon(free_space_polygon)


def extract_voxels_hashmap(group_clouds):
    """
    Create a spatial hash map of voxels occupied by the point cloud.
    """
    hash_map = defaultdict(list)
    for _, points in group_clouds:
        for point in points:
            voxel_key = tuple((point[:3] // const.g_grid_size).astype(int))
            hash_map[voxel_key].append(point)

    return hash_map

def extract_voxels_hashmap_points(grid_size, points):
    hash_map = defaultdict(list)
    for point in points:
        voxel_key = tuple((point[:3] // grid_size).astype(int))
        hash_map[voxel_key].append(point)

    return hash_map

def extract_voxels_keys_points(points, grid_size=const.g_grid_size):
    voxel_keys = (points[:, :3] // grid_size).astype(int)
    voxel_keys = np.unique(voxel_keys, axis=0)
    return voxel_keys

def extract_selected_points(group_clouds, selected_group):
    selected_points = []
    for category, points in group_clouds:
        if category in selected_group:
            selected_points.append(points)
    return np.vstack(selected_points) if selected_points else np.empty((0, 6))

def extract_selected_voxels_keys(group_clouds, selected_group):
    voxel_keys = [] 

    for category, points in group_clouds:
        if category in selected_group:
            for point in points:
                voxel_key = (point[:3] // const.g_grid_size).astype(int)
                voxel_keys.append(voxel_key)

    # Combine all keys into a single NumPy array
    if voxel_keys:
        voxel_keys = np.vstack(voxel_keys)
        voxel_keys = np.unique(voxel_keys, axis=0)  # Remove duplicates
    else:
        voxel_keys = np.empty((0, 3), dtype=int)    # Return empty array if no keys

    return voxel_keys

def extract_intersected_voxels(voxel_keys_1, voxel_keys_2):
    if voxel_keys_1.ndim == 2 and voxel_keys_2.ndim == 2:
        voxel_keys_1_view = voxel_keys_1.view([('', voxel_keys_1.dtype)] * voxel_keys_1.shape[1])
        voxel_keys_2_view = voxel_keys_2.view([('', voxel_keys_2.dtype)] * voxel_keys_2.shape[1])

        intersected_voxels = np.intersect1d(voxel_keys_1_view, voxel_keys_2_view)
        intersected_voxels = intersected_voxels.view(voxel_keys_1.dtype).reshape(-1, voxel_keys_1.shape[1])
        intersected_voxels = np.array(intersected_voxels, dtype=np.int64)
        
    else:
        raise ValueError("Voxel arrays must be 2D(row, column) for row-wise comparison.")
    
    return intersected_voxels

def filter_floor_voxels(floor_voxels, feature_voxels):
    if floor_voxels.size == 0 or feature_voxels.size == 0:
        return floor_voxels

    # Get feature height range
    feat_min_y = np.min(feature_voxels[:, 1])
    feat_max_y = np.max(feature_voxels[:, 1])

    # Calculate height threshold for features
    height_threshold = feat_min_y + (feat_max_y - feat_min_y) * const.g_height_threshold
    
    # Get features below threshold
    features_below_threshold = feature_voxels[feature_voxels[:, 1] <= height_threshold]
    
    # Find overlapping x,z coordinates with filtered features
    overlap_mask = np.ones(len(floor_voxels), dtype=bool)
    for i, floor_point in enumerate(floor_voxels[:, [0, 2]]):
        # Check if this x,z coordinate exists in filtered feature voxels
        matches = np.all(features_below_threshold[:, [0, 2]] == floor_point, axis=1)
        if np.any(matches):
            overlap_mask[i] = False

    return floor_voxels[overlap_mask]