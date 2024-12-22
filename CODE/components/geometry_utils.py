import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import make_valid
from collections import defaultdict
import components.constants as const


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
    """
    Apply a transformation to a point cloud based on rotation and translation.
    """
    theta, tx, tz = transformation
    centroid_x, centroid_z = centroid

    # Convert theta to radians for computation
    theta_rad = np.radians(theta)

    # Step 1: Translate to origin (compute centroid of X and Z)
    transformed_cloud = []
    for category_name, points in cloud:
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


def extract_free_space_polygon(cloud):
    """
    Extract a polygon representing the free space in the point cloud.
    """
    free_space_polygon = None
    excluded_polygons = []

    for category_name, points in cloud:
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


def extract_voxels_hashmap(cloud):
    """
    Create a spatial hash map of voxels occupied by the point cloud.
    """
    hash_map = defaultdict(list)
    for category_name, points in cloud:
        for point in points:
            voxel_key = tuple((point // const.g_grid_size).astype(int))
            hash_map[voxel_key].append(point)

    return hash_map
