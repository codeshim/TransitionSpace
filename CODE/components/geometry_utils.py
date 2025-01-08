import numpy as np
import open3d as o3d
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from collections import defaultdict
import components.constants as const

# def map_group_clouds_to_all_points(group_clouds):
#     """
#     group_clouds
#     {{{"category"}, {X, Y, Z, R, G, B}}, ...}}
#     ->
#     all_points
#     {X, Y, Z, R, G, B, C}
#     """
#     all_points = np.vstack([
#                      np.hstack([points, np.full((points.shape[0], 1), const.CATEGORY_MAPPING[category_name])])
#                      for category_name, points in group_clouds])

#     return all_points

# def map_all_points_to_group_clouds(all_points):
#     """   
#     all_points
#     {X, Y, Z, R, G, B, C}
#     ->
#     group_clouds
#     {{{"category"}, {X, Y, Z, R, G, B}}, ...}}
#     """
#     group_clouds = []
#     category_ids = np.unique(all_points[:, -1])
#     id_to_category = {v: k for k, v in const.CATEGORY_MAPPING.items()}
    
#     for category_id in category_ids:
#         category_points = all_points[all_points[:, -1] == category_id]
#         category_points = category_points[:, :-1]
#         category_name = id_to_category[category_id]
#         group_clouds[category_name] = category_points

#     return group_clouds


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
    # """
    # Extract unique voxel keys (x, y, z) occupied by the point cloud,
    # filtering only points belonging to categories in the selected_group.
    # Returns a NumPy array of unique voxel keys.
    # """
    # voxel_keys = []  # Collect voxel keys as a list
    # for category, points in group_clouds:
    #     # Check if the category is in the selected group
    #     #print(f"category: {category}")
    #     #print(f"points: {points}, points.shape: {points.shape}")
    #     """
    #     category: chair
    #     points: [[  1.86225      0.587        1.891       96.         102.
    #       106.        ]
    #      [  1.86373333   0.59446667   1.86233333 104.         109.
    #       116.        ]
    #      [  1.848        0.797        2.187      115.         117.
    #        96.        ]
    #      ...
    #      [  1.84118519   0.21211111   2.01640741  51.          53.
    #        45.        ]
    #      [  1.65233333   0.51591667   2.09158333  70.          71.
    #        66.        ]
    #      [  1.69814286   0.59910714   2.1155      73.          77.
    #        84.        ]], points.shape: (618, 6)
    #     """
    #     if category in selected_group:
    #         # Calculate voxel keys for all points in this category
    #         for point in points:
    #             keys = (point[:3] // grid_size).astype(int)
    #             voxel_keys.append(keys)

    # # Combine all keys into a single NumPy array
    # if voxel_keys:
    #     voxel_keys = np.vstack(voxel_keys)
    #     voxel_keys = np.unique(voxel_keys, axis=0)  # Remove duplicates
    # else:
    #     voxel_keys = np.empty((0, 3), dtype=int)    # Return empty array if no keys

    # return voxel_keys
