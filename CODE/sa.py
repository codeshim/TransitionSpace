import argparse
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely import affinity
from S3DIS_to_json import jsonl_to_array
from optimization.sa_threeplus import simulated_annealing
import open3d as o3d

CATEGORY_MAPPING = {
    "ceiling": 1,
    "floor": 2,
    "wall": 3,
    "beam": 4,
    "column": 5,
    "window": 6,
    "door": 7,
    "table": 8,
    "chair": 9,
    "sofa": 10,
    "bookcase": 11,
    "board": 12,
    "clutter": 0
}

# Target function for simulated annealing
def target_function(polygonList, variables):
    """
    Calculate the negative overlap area between local and transformed remote polygons.
    """
    local_polygon = polygonList[0]
    remote_polygon = polygonList[1]

    # Apply transformations to the remote polygon
    for i, var_set in enumerate(variables):
        tx, tz, theta = var_set[0], var_set[1], var_set[2]
        remote_polygon = affinity.translate(remote_polygon, xoff=tx, yoff=tz)
        remote_polygon = affinity.rotate(remote_polygon, theta, origin="centroid")
        polygonList[1] = remote_polygon

    # Calculate the overlap area
    overlap_area = local_polygon.intersection(remote_polygon).area
    return -overlap_area  # Negative for minimization

# Function to extract the ceiling polygon
def extract_ceiling_polygon(group_clouds, excluded_categories):
    ceiling_polygon = None
    excluded_polygons = []

    for category_name, points in group_clouds:
        points_2d = points[:, [0, 2]]  # Project to X-Z plane
        if category_name == "ceiling":
            ceiling_polygon = Polygon(points_2d).convex_hull
        elif category_name in excluded_categories:
            excluded_polygons.append(Polygon(points_2d).convex_hull)

    if not ceiling_polygon:
        raise ValueError("Ceiling category not found in the provided point cloud data.")

    if excluded_polygons:
        union_excluded = unary_union(excluded_polygons)
        ceiling_polygon = ceiling_polygon.difference(union_excluded)

    return ceiling_polygon

# Function to plot polygons
def plot_polygon(polygon, color, label=None):
    if isinstance(polygon, Polygon):
        x, y = polygon.exterior.xy
        plt.plot(x, y, color=color, label=label)
    elif isinstance(polygon, MultiPolygon):
        for i, sub_polygon in enumerate(polygon.geoms):
            label = label if i == 0 else None  # Add label only to the first plot
            x, y = sub_polygon.exterior.xy
            plt.plot(x, y, color=color, label=label)

# Main simulation
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote JSONL file.")
    parser.add_argument("--excluded_categories", type=str, nargs='+', default=[
        "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
    ], help="Categories to exclude from the ceiling area.")

    args = parser.parse_args()

    # Load point clouds
    local_clouds = jsonl_to_array(args.loc)
    remote_clouds = jsonl_to_array(args.rmt)

    # Extract polygons
    local_polygon = extract_ceiling_polygon(local_clouds, args.excluded_categories)
    remote_polygon = extract_ceiling_polygon(remote_clouds, args.excluded_categories)

    # Setup polygons for optimization
    polygonList = [local_polygon, remote_polygon]

    # Simulated Annealing Parameters
    min_values = [-5, -5, -180]  # Translation (x, z) and rotation bounds
    max_values = [5, 5, 180]
    sa_params = {
        'min_values': min_values,
        'max_values': max_values,
        'min': 0.0,
        'max': 1.0,
        'initial_temperature': 1.0,
        'temperature_iterations': 100,
        'final_temperature': 0.0001,
        'alpha': 0.95,
        'polygonList': polygonList,
        'target_function': target_function,
        'verbose': True
    }

    # Run Simulated Annealing
    best_area, optimized_polygons, optimized_theta_sum = simulated_annealing(**sa_params)

    print("Best Overlap Area: ", -best_area)
    print("Optimized Transformation Parameters: ", optimized_theta_sum)

    # Visualization
    plt.figure(figsize=(8, 8))
    plot_polygon(local_polygon, color="blue", label="Local Ceiling")
    plot_polygon(optimized_polygons[1], color="red", label="Optimized Remote Ceiling")
    plt.legend()
    plt.title("Optimized Overlap Area")
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.axis("equal")
    plt.grid(True)
    plt.show()

    # Open3D visualization
    local_pcd = o3d.geometry.PointCloud()
    local_pcd.points = o3d.utility.Vector3dVector(np.array([pt[:3] for _, pts in local_clouds for pt in pts]))
    local_colors = np.array([CATEGORY_MAPPING[cat] for cat, pts in local_clouds for _ in pts]) / len(CATEGORY_MAPPING)
    local_pcd.colors = o3d.utility.Vector3dVector(np.repeat(local_colors[:, np.newaxis], 3, axis=1))

    remote_pcd = o3d.geometry.PointCloud()
    remote_pcd.points = o3d.utility.Vector3dVector(np.array([pt[:3] for _, pts in remote_clouds for pt in pts]))
    remote_colors = np.array([CATEGORY_MAPPING[cat] for cat, pts in remote_clouds for _ in pts]) / len(CATEGORY_MAPPING)
    remote_pcd.colors = o3d.utility.Vector3dVector(np.repeat(remote_colors[:, np.newaxis], 3, axis=1))

    # Apply optimized transformation to remote point cloud
    tx, tz, theta = optimized_theta_sum[0], optimized_theta_sum[1], optimized_theta_sum[2]
    transformation_matrix = np.eye(4)
    transformation_matrix[0, 3] = tx
    transformation_matrix[2, 3] = tz
    transformation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle([0, np.radians(theta), 0])

    remote_pcd.transform(transformation_matrix)

    # Visualize in Open3D
    o3d.visualization.draw_geometries([local_pcd, remote_pcd], window_name="Optimized Point Clouds", width=800, height=600)
