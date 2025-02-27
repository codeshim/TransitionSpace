import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import components.geometry_utils as utils
import components.constants as const
import components.record as rec
from components.objective_functions import calculate_shared_space

# Function to calculate voxel center from cell key
def voxel_centers(cell_key):
    # Calculate the center of the voxel based on its grid position
    return np.array(cell_key) * const.g_grid_size + (const.g_grid_size / 2.0)

def create_voxel_wireframe(center):
    # Define the 8 corners of the voxel
    half_size = const.g_grid_size / 2.0
    corners = [
        center + np.array([half_size, half_size, half_size]),
        center + np.array([half_size, half_size, -half_size]),
        center + np.array([half_size, -half_size, half_size]),
        center + np.array([half_size, -half_size, -half_size]),
        center + np.array([-half_size, half_size, half_size]),
        center + np.array([-half_size, half_size, -half_size]),
        center + np.array([-half_size, -half_size, half_size]),
        center + np.array([-half_size, -half_size, -half_size])
    ]

    # Define the 12 edges connecting the corners
    lines = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7]
    ]

    # Create a LineSet object from the corners and lines
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set

def draw_voxel_wireframe(keys, grid_size, colors):
    # Define the 12 edges connecting the corners
    lines = [
        [0, 1], [0, 2], [0, 4],
        [1, 3], [1, 5],
        [2, 3], [2, 6],
        [3, 7],
        [4, 5], [4, 6],
        [5, 7],
        [6, 7]
    ]
    line_sets = []
    for key in keys:
        center = np.array(key) * grid_size + (grid_size / 2.0)
        half_size = grid_size / 2.0
        corners = [
            center + np.array([half_size, half_size, half_size]),
            center + np.array([half_size, half_size, -half_size]),
            center + np.array([half_size, -half_size, half_size]),
            center + np.array([half_size, -half_size, -half_size]),
            center + np.array([-half_size, half_size, half_size]),
            center + np.array([-half_size, half_size, -half_size]),
            center + np.array([-half_size, -half_size, half_size]),
            center + np.array([-half_size, -half_size, -half_size])
        ]      

        # Create a LineSet object from the corners and lines
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(colors)
        line_sets.append(line_set)

    return line_sets

def draw_voxel_box(keys, grid_size, colors):
    boxes = []
    for key in keys:
        center = np.array(key) * grid_size + (grid_size / 2.0)
        box = o3d.geometry.TriangleMesh.create_box(width=grid_size, height=grid_size, depth=grid_size)
        box.translate(center - (grid_size / 2.0))  # Center the box
        box.paint_uniform_color(colors)
        boxes.append(box)
    
    return boxes

def visualize_and_record_pareto_front(record=True):
    print(f"tr: {const.g_best_tr}, obj1: {const.g_best_obj1}, obj2: {const.g_best_obj2}")
    # Extract the transformation from pereto_set[0]
    transformation = tuple(const.g_best_tr)

    # Apply transformation to the remote cloud
    transformed_remote_cloud = utils.apply_points_transformation(const.g_remote_cloud_origin, const.g_remote_centroid, transformation)

    # Convert point clouds to Open3D format
    local_all_cloud_points = np.vstack([points for _, points in const.g_local_cloud_origin])
    local_cloud_o3d = o3d.geometry.PointCloud()
    local_cloud_o3d.points = o3d.utility.Vector3dVector(local_all_cloud_points[:, :3])
    local_cloud_o3d.colors = o3d.utility.Vector3dVector(local_all_cloud_points[:, 3:6] / 255.0)

    remote_all_cloud_points = np.vstack([points for _, points in transformed_remote_cloud])
    remote_cloud_o3d = o3d.geometry.PointCloud()
    remote_cloud_o3d.points = o3d.utility.Vector3dVector(remote_all_cloud_points[:, :3])
    remote_cloud_o3d.colors = o3d.utility.Vector3dVector(remote_all_cloud_points[:, 3:6] / 255.0)

    shared_meshes = []
    vis_sets = []

    if (not const.g_isallvoxel):
        # Visualize shared polygon     
        shared_2d_coords = []  # Store 2D (x, z) coordinates for matplotlib visualization

        transformed_remote_polygon = utils.extract_free_space_polygon(transformed_remote_cloud)
        _, shared_space = calculate_shared_space(transformed_remote_polygon, transformation)

        # Check if shared_space is a single Polygon or MultiPolygon
        if shared_space is not None:
            # Check if shared_space is a single Polygon or MultiPolygon
            if shared_space.geom_type == "Polygon":
                polygons = [shared_space]  # Treat it as a single-element list for uniform processing
            elif shared_space.geom_type == "MultiPolygon":
                polygons = list(shared_space.geoms)  # Extract geometries from MultiPolygon
            else:
                polygons = []  # Handle unexpected types (e.g., empty geometry)    

            for polygon in polygons:
                if polygon.is_empty:
                    continue
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
        else:
            print("Warning: No valid shared space found")

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

        if record:
            rec.save_plt_fig("2D")
        
        plt.show()
    else:
        vis_strt_keys = const.g_overlap_strt_voxels[transformation]
        vis_sets.extend(visualize_filled_voxels(vis_strt_keys, color=[0.0, 0.0, 1.0]))
        
    # Visualize keys
    if (const.g_ismulitobj):
        vis_feat_keys = const.g_overlap_feat_voxels[transformation]
        vis_sets.extend(visualize_filled_voxels(vis_feat_keys, color=[1.0, 0.0, 0.0]))

    # Open3D visualization setup
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if record == True:
        vis.register_key_callback(ord('P'), rec.capture_screenshot)
    vis.create_window()
    vis.add_geometry(local_cloud_o3d)
    vis.add_geometry(remote_cloud_o3d)
    if (not const.g_isallvoxel):
        vis.add_geometry(shared_meshes)
    for vis_item in vis_sets:
        vis.add_geometry(vis_item)
    vis.run()
    vis.destroy_window()

    # Generate a sequence of generations
    generations = list(range(1, len(const.g_best_obj1_list) + 1))

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, const.g_best_obj1_list, '-o', label='Objective 1')
    plt.plot(generations, const.g_best_obj2_list, '-s', label='Objective 2')

    # Add labels, title, and legend
    plt.xlabel('Generation')
    plt.ylabel('Objective Values')
    plt.title('Best Objectives Across Generations')
    plt.legend(loc='best')
    plt.grid(True)

    if record:
        rec.save_plt_fig("gen")
    # Show the plot
    plt.show()
    
    if record:
        rec.save_report()

def visualize_wireframe_voxels(voxel_keys, color=[1.0, 0.0, 0.0]):
    vis_sets = []
    if voxel_keys is not None and len(voxel_keys) > 0:
        for key in voxel_keys:
            voxel_center = voxel_centers(key)
            line_set = create_voxel_wireframe(voxel_center)
            line_set.paint_uniform_color(color)
            vis_sets.append(line_set)

    return vis_sets

def visualize_filled_voxels(voxel_keys, color=[1.0, 0.0, 0.0]):
    vis_sets = []
    if voxel_keys is not None and len(voxel_keys) > 0:
        for key in voxel_keys:
            voxel_center = voxel_centers(key)
            voxel = o3d.geometry.TriangleMesh.create_box(width=const.g_grid_size,
                                                         height=const.g_grid_size,
                                                         depth=const.g_grid_size)
            voxel.translate(voxel_center - (const.g_grid_size / 2.0))  # Center the box
            voxel.paint_uniform_color(color)  # Red color
            vis_sets.append(voxel)

    return vis_sets

# def visualize_pareto_front_old(pereto_set):
#     """
#     Visualize the best result (pereto_set[0]) with Open3D and matplotlib.
#     """
#     # Extract the transformation from pereto_set[0]
#     theta, tx, tz = pereto_set[:3]
#     transformation = (theta, tx, tz)

#     # Apply transformation to the remote cloud
#     transformed_remote_cloud = utils.apply_points_transformation(const.g_remote_cloud, const.g_remote_centroid, transformation)

#     # Convert point clouds to Open3D format
#     local_all_cloud_points = np.vstack([points for _, points in const.g_local_cloud])
#     remote_all_cloud_points = np.vstack([points for _, points in transformed_remote_cloud])

#     print(f"local_all_cloud_points:{local_all_cloud_points[0]}")
#     print(f"remote_all_cloud_points:{remote_all_cloud_points[0]}")

#     local_cloud_o3d = o3d.geometry.PointCloud()
#     local_cloud_o3d.points = o3d.utility.Vector3dVector(local_all_cloud_points[:, :3])

#     remote_cloud_o3d = o3d.geometry.PointCloud()
#     remote_cloud_o3d.points = o3d.utility.Vector3dVector(remote_all_cloud_points[:, :3])

#     # Color the clouds for differentiation
#     local_cloud_o3d.colors = o3d.utility.Vector3dVector(local_all_cloud_points[:, 3:6] / 255.0)
#     remote_cloud_o3d.colors = o3d.utility.Vector3dVector(remote_all_cloud_points[:, 3:6] / 255.0)

#     # Visualize voxel loop as line segments
#     # voxel_keys = const.g_voxel_loops[transformation]
#     # line_sets = []
#     # transformed_remote_voxels = utils.extract_voxels_hashmap(transformed_remote_cloud)
#     # all_voxel_keys = set(const.g_local_voxels.keys()).intersection(set(transformed_remote_voxels.keys()))
#     # for key in all_voxel_keys:
#     #     voxel_center = voxel_centers(key)
#     #     line_set = create_voxel_wireframe(voxel_center)
#     #     line_set.paint_uniform_color([1.0, 0.0, 0.0])
#     #     line_sets.append(line_set)

#     # for key in voxel_keys:
#     #     voxel_center = voxel_centers(key)
#     #     voxel = o3d.geometry.TriangleMesh.create_box(width=const.g_grid_size,
#     #                                                  height=const.g_grid_size,
#     #                                                  depth=const.g_grid_size)
#     #     voxel.translate(voxel_center - (const.g_grid_size / 2.0))  # Center the box
#     #     voxel.paint_uniform_color([1.0, 0.0, 0.0])  # Red color
#     #     line_sets.append(voxel)

#     # print(f"voxel_keys num: {len(voxel_keys)}, tr: {pereto_set[0]}, {pereto_set[1]}, {pereto_set[2]}, obj1: {pereto_set[3]}, obj2: {pereto_set[4]}")

#     # for voxel in voxel_keys:
#     #     center = np.array(voxel) * const.g_grid_size  # Convert voxel keys to real-world positions
#     #     voxel_lines.append(center)

#     # Convert to Open3D lines
#     # line_set.points = o3d.utility.Vector3dVector(voxel_lines)
#     # line_set.lines = o3d.utility.Vector2iVector(
#     #     [[i, i + 1] for i in range(len(voxel_lines) - 1)]
#     # )

#     # Visualize shared polygon
#     shared_polygons = const.g_shared_polygon[transformation]
#     shared_meshes = []
#     shared_2d_coords = []  # Store 2D (x, z) coordinates for matplotlib visualization

#     for polygon in shared_polygons:
#         if polygon.is_empty:
#             continue
#         if polygon.geom_type == "Polygon":
#             coords = np.array(polygon.exterior.coords)
#             shared_2d_coords.append(coords)
#             # Convert 2D to 3D
#             coords_3d = np.array([[x, 0.05, z] for x, z in coords])
#             shared_mesh = o3d.geometry.LineSet()
#             shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
#             shared_mesh.lines = o3d.utility.Vector2iVector(
#                 [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
#             )
#             shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
#             shared_meshes.append(shared_mesh)
#         elif polygon.geom_type == "MultiPolygon":
#             for sub_polygon in polygon.geoms:
#                 coords = np.array(sub_polygon.exterior.coords)
#                 shared_2d_coords.append(coords)
#                 # Convert 2D to 3D
#                 coords_3d = np.array([[x, 0.05, z] for x, z in coords])
#                 shared_mesh = o3d.geometry.LineSet()
#                 shared_mesh.points = o3d.utility.Vector3dVector(coords_3d)
#                 shared_mesh.lines = o3d.utility.Vector2iVector(
#                     [[i, i + 1] for i in range(len(coords_3d) - 1)] + [[len(coords_3d) - 1, 0]]
#                 )
#                 shared_mesh.paint_uniform_color([1, 0, 1])  # Magenta for shared polygon
#                 shared_meshes.append(shared_mesh)

#     # Visualize with matplotlib (2D projection)
#     plt.figure(figsize=(10, 10))
#     plt.scatter(local_all_cloud_points[:, 0], local_all_cloud_points[:, 2], c=local_all_cloud_points[:, 3:6] / 255.0, s=1, label="Local Cloud")
#     plt.scatter(remote_all_cloud_points[:, 0], remote_all_cloud_points[:, 2], c=remote_all_cloud_points[:, 3:6] / 255.0, s=1, label="Transformed Remote Cloud")

#     for coords in shared_2d_coords:
#         plt.plot(coords[:, 0], coords[:, 1], c='magenta', label="Shared Polygon")

#     plt.xlabel("X")
#     plt.ylabel("Z")
#     plt.legend()
#     plt.title("2D Visualization of Clouds and Shared Polygon")
#     plt.show()

#     # Visualize all components with Open3D (3D visualization)
#     o3d.visualization.draw_geometries(
#         [local_cloud_o3d, remote_cloud_o3d] + line_sets + shared_meshes
#     )

#     # Generate a sequence of generations
#     generations = list(range(1, len(const.g_best_obj1) + 1))

#     # Create the plot
#     plt.figure(figsize=(10, 6))
#     plt.plot(generations, const.g_best_obj1, '-o', label='Objective 1')
#     plt.plot(generations, const.g_best_obj2, '-s', label='Objective 2')

#     # Add labels, title, and legend
#     plt.xlabel('Generation')
#     plt.ylabel('Objective Values')
#     plt.title('Best Objectives Across Generations')
#     plt.legend(loc='best')
#     plt.grid(True)

#     # Show the plot
#     plt.show()