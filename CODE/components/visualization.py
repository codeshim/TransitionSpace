import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import components.geometry_utils as utils
import components.constants as const

def visualize_pareto_front(pereto_set):
    """
    Visualize the best result (pereto_set[0]) with Open3D and matplotlib.
    """
    # Extract the transformation from pereto_set[0]
    theta, tx, tz = pereto_set[:3]
    transformation = (theta, tx, tz)

    # Apply transformation to the remote cloud
    transformed_remote_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, transformation)

    # Convert point clouds to Open3D format
    local_all_cloud_points = np.vstack([points for _, points in const.g_local_cloud])
    remote_all_cloud_points = np.vstack([points for _, points in transformed_remote_cloud])

    print(f"local_all_cloud_points:{local_all_cloud_points[0]}")
    print(f"remote_all_cloud_points:{remote_all_cloud_points[0]}")

    local_cloud_o3d = o3d.geometry.PointCloud()
    local_cloud_o3d.points = o3d.utility.Vector3dVector(local_all_cloud_points[:, :3])

    remote_cloud_o3d = o3d.geometry.PointCloud()
    remote_cloud_o3d.points = o3d.utility.Vector3dVector(remote_all_cloud_points[:, :3])

    # Color the clouds for differentiation
    local_cloud_o3d.colors = o3d.utility.Vector3dVector(local_all_cloud_points[:, 3:6] / 255.0)
    remote_cloud_o3d.colors = o3d.utility.Vector3dVector(remote_all_cloud_points[:, 3:6] / 255.0)

    # Visualize voxel loop as line segments
    voxel_keys = const.g_voxel_loops[transformation]
    voxel_lines = []
    line_set = o3d.geometry.LineSet()

    for voxel in voxel_keys:
        center = np.array(voxel) * const.g_grid_size  # Convert voxel keys to real-world positions
        voxel_lines.append(center)

    # Convert to Open3D lines
    line_set.points = o3d.utility.Vector3dVector(voxel_lines)
    line_set.lines = o3d.utility.Vector2iVector(
        [[i, i + 1] for i in range(len(voxel_lines) - 1)]
    )
    line_set.paint_uniform_color([0, 1, 0])  # Green for voxel loops

    # Visualize shared polygon
    shared_polygons = const.g_shared_polygon[transformation]
    shared_meshes = []
    shared_2d_coords = []  # Store 2D (x, z) coordinates for matplotlib visualization

    for polygon in shared_polygons:
        if polygon.is_empty:
            continue
        if polygon.geom_type == "Polygon":
            coords = np.array(polygon.exterior.coords)
            shared_2d_coords.append(coords)
            # Convert 2D to 3D
            coords_3d = np.array([[x, 0.05, z] for x, z in coords])
            shared_mesh = o3d.geometry.LineSet()
            shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
            shared_mesh.lines = o3d.utility.Vector2iVector(
                [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
            )
            shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
            shared_meshes.append(shared_mesh)
        elif polygon.geom_type == "MultiPolygon":
            for sub_polygon in polygon.geoms:
                coords = np.array(sub_polygon.exterior.coords)
                shared_2d_coords.append(coords)
                # Convert 2D to 3D
                coords_3d = np.array([[x, 0.05, z] for x, z in coords])
                shared_mesh = o3d.geometry.LineSet()
                shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
                shared_mesh.lines = o3d.utility.Vector2iVector(
                    [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
                )
                shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
                shared_meshes.append(shared_mesh)

    # Visualize with matplotlib (2D projection)
    plt.figure(figsize=(10, 10))
    plt.scatter(local_all_cloud_points[:, 0], local_all_cloud_points[:, 2], c=local_all_cloud_points[:, 3:6] / 255.0, s=1, label="Local Cloud")
    plt.scatter(remote_all_cloud_points[:, 0], remote_all_cloud_points[:, 2], c=remote_all_cloud_points[:, 3:6] / 255.0, s=1, label="Transformed Remote Cloud")

    for coords in shared_2d_coords:
        plt.plot(coords[:, 0], coords[:, 1], c='magenta', label="Shared Polygon")

    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.title("2D Visualization of Clouds and Shared Polygon")
    plt.show()

    # Visualize all components with Open3D (3D visualization)
    o3d.visualization.draw_geometries(
        [local_cloud_o3d, remote_cloud_o3d, line_set] + shared_meshes
    )