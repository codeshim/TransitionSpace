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
    for category_name, points in group_clouds:
        # Downsampling
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
        original_count = len(pcd.points)
        down_pcd = pcd.voxel_down_sample(voxel_size)
        downsampled_count = len(down_pcd.points)

        # Print detailed information
        reduced_count = original_count - downsampled_count
        reduction_percentage = (reduced_count / original_count) * 100 if original_count > 0 else 0
        print(f"Category: {category_name}")
        print(f"  Original points: {original_count}")
        print(f"  Downsampled points: {downsampled_count}")
        print(f"  Points reduced: {reduced_count} ({reduction_percentage:.2f}%)")
        
        # Append back to the result dictionary
        down_points = np.asarray(down_pcd.points)
        down_colors = (np.asarray(down_pcd.colors) * 255).astype(np.uint8)
        downsampled_data = np.hstack((down_points, down_colors))
        down_group_clouds.append((category_name, downsampled_data))

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
    for category_name, points in group_clouds:
        for point in points:
            voxel_key = tuple((point // const.g_grid_size).astype(int))
            hash_map[voxel_key].append(point)

    return hash_map
