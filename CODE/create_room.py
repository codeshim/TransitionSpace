import open3d as o3d
import numpy as np
import components.S3DIS_to_json as s3dis
import components.constants as const
import components.geometry_utils as utils
import components.visualization as vis


def create_cuboid_point_cloud(width, height, depth, density):
    """
    Create a point cloud representing the surface of a cuboid.

    :param width: Width of the cuboid (X-axis length).
    :param height: Height of the cuboid (Y-axis length).
    :param depth: Depth of the cuboid (Z-axis length).
    :param density: Number of points per unit length on each face.
    :return: Open3D PointCloud object.
    """
    # Generate points for each face
    x = np.linspace(-width / 2, width / 2, int(density * width))
    y = np.linspace(-height / 2, height / 2, int(density * height))
    z = np.linspace(-depth / 2, depth / 2, int(density * depth))

    # Top and bottom faces
    top = np.array(np.meshgrid(x, z)).T.reshape(-1, 2)
    top = np.c_[top[:, 0], np.full_like(top[:, 0], height / 2), top[:, 1]]
    bottom = np.array(np.meshgrid(x, z)).T.reshape(-1, 2)
    bottom = np.c_[bottom[:, 0], np.full_like(bottom[:, 0], -height / 2), bottom[:, 1]]

    # Front and back faces
    front = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    front = np.c_[front[:, 0], front[:, 1], np.full_like(front[:, 0], depth / 2)]
    back = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    back = np.c_[back[:, 0], front[:, 1], np.full_like(front[:, 0], -depth / 2)]

    # Left and right faces
    left = np.array(np.meshgrid(z, y)).T.reshape(-1, 2)
    left = np.c_[np.full_like(left[:, 0], -width / 2), left[:, 1], left[:, 0]]
    right = np.array(np.meshgrid(z, y)).T.reshape(-1, 2)
    right = np.c_[np.full_like(left[:, 0], width / 2), right[:, 1], right[:, 0]]

    # Combine all points
    points = np.vstack((top, bottom, front, back, left, right))

    # Create Open3D PointCloud
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    return points

def extract_excluded_keys(keys, overlapping_keys):
    # Step 1: Group by x and find min/max z and y
    bound_xz = set()
    bound_y = set()

    # Extract the ranges for x, z, and y
    min_x = min(x for x, _, _ in overlapping_keys)
    max_x = max(x for x, _, _ in overlapping_keys)
    min_z = min(z for _, _, z in overlapping_keys)
    max_z = max(z for _, _, z in overlapping_keys)
    min_y = min(y for _, y, _ in overlapping_keys)
    max_y = max(y for _, y, _ in overlapping_keys)

    # Add all (x, z) within the bounds
    for x in range(min_x, max_x + 1):
        for z in range(min_z, max_z + 1):
            bound_xz.add((x, z))

    # Add all y within the bounds
    for y in range(min_y, max_y + 1):
        bound_y.add(y)

    # Step 2: Select all y values that exist in keys for (x, z)
    excluded_keys = set()
    for x, z in bound_xz:
        # Filter y values that are explicitly in keys
        all_ys = [key[1] for key in keys.keys() if key[0] == x and key[2] == z]
        for y in all_ys:  # Only include valid y values
            if y in bound_y:
                key = (x, y, z)
                if key not in overlapping_keys:
                    excluded_keys.add(key)

    return excluded_keys

def filter_points_by_excluded_keys(points, excluded_keys, grid_size):
    """
    Filter points to exclude those that belong to the specified excluded voxel keys.

    :param points: Array of points to filter.
    :param excluded_keys: Set of voxel keys to exclude.
    :param grid_size: Size of the grid for voxelization.
    :return: Filtered array of points.
    """
    filtered_points = []
    for point in points:
        voxel_key = tuple((point[:3] // grid_size).astype(int))
        if voxel_key not in excluded_keys:
            filtered_points.append(point)
    return np.array(filtered_points)

def create_mesh_poisson(points, depth=8):
    """
    Create a triangular mesh from a set of points using Poisson Surface Reconstruction.

    :param points: NumPy array of points to create the mesh from.
    :param depth: Depth parameter for Poisson reconstruction. Higher values result in finer meshes.
    :return: Open3D TriangleMesh object.
    """
    # Convert points to Open3D PointCloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Estimate normals for the point cloud
    point_cloud.estimate_normals()
    point_cloud.orient_normals_consistent_tangent_plane(30)

    # Perform Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=depth
    )

    # Optionally filter low-density vertices
    densities = np.asarray(densities)
    vertices_to_remove = densities < np.quantile(densities, 0.01)  # Remove vertices with lowest density
    mesh.remove_vertices_by_mask(vertices_to_remove)

    return mesh

def color_mesh(mesh, color):
    """
    Apply a uniform color to all vertices of a mesh.

    :param mesh: Open3D TriangleMesh object.
    :param color: List of RGB values in the range [0, 1].
    """
    colors = np.tile(color, (len(mesh.vertices), 1))  # Create a color array for all vertices
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

def enable_double_sided_rendering(mesh):
    """
    Modify a mesh to support double-sided rendering by duplicating triangles with reversed normals.

    :param mesh: Open3D TriangleMesh object.
    :return: Modified Open3D TriangleMesh with double-sided rendering.
    """
    # Reverse the triangles
    reversed_triangles = np.asarray(mesh.triangles)[:, ::-1]  # Reverse triangle indices
    reversed_normals = -np.asarray(mesh.vertex_normals)       # Reverse normals

    # Create a new mesh with the reversed triangles and normals
    reversed_mesh = o3d.geometry.TriangleMesh()
    reversed_mesh.vertices = mesh.vertices
    reversed_mesh.triangles = o3d.utility.Vector3iVector(reversed_triangles)
    reversed_mesh.vertex_normals = o3d.utility.Vector3dVector(reversed_normals)
    if mesh.has_vertex_colors():
        reversed_mesh.vertex_colors = mesh.vertex_colors

    # Combine the original and reversed meshes
    combined_mesh = mesh + reversed_mesh

    return combined_mesh




density_grid_size = 0.02
grid_size = 0.15

loc_color = [255 / 255, 113 / 255, 91 / 255]
rmt_color = [34 / 255, 137 / 255, 221 / 255]
overlap_color = [124 / 255, 13 / 255, 198 / 255]

# ========================================= Simple cuboid test  ========================================= 

# # Parameters for the cuboid
# loc_width = 6.0
# loc_depth = 6.0
# loc_height = 3.0
# loc_density = 10

# # Generate the cuboid point cloud
# loc_cuboid_points = create_cuboid_point_cloud(loc_width, loc_height, loc_depth, loc_density)

# loc_anchor_mat = s3dis.compute_anchor_mat(loc_cuboid_points)
# loc_cuboid_points = s3dis.transform_points_by_anchor(loc_anchor_mat, loc_cuboid_points)
# #cuboid_points = s3dis.transform_to_left_handed_y_up(cuboid_points)

# loc_cuboid_pcd = o3d.geometry.PointCloud()
# loc_cuboid_pcd.points = o3d.utility.Vector3dVector(loc_cuboid_points)
# #loc_color = [141 / 255, 90 / 255, 151 / 255]
# loc_color = [255 / 255, 113 / 255, 91 / 255]
# loc_cuboid_pcd.colors = o3d.utility.Vector3dVector([loc_color] * len(loc_cuboid_pcd.points))
# loc_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, loc_cuboid_points)

# # Visualize
# loc_voxel_wireframe = vis.draw_voxel_wireframe(loc_voxel_keys, grid_size, loc_color)

# # Parameters for the cuboid
# rmt_width = 3.0
# rmt_depth = 3.0
# rmt_height = 3.0
# rmt_density = 10
15
# rmt_cuboid_points = create_cuboid_point_cloud(rmt_width, rmt_height, rmt_depth, rmt_density)

# rmt_anchor_mat = s3dis.compute_anchor_mat(rmt_cuboid_points)
# rmt_cuboid_points = s3dis.transform_points_by_anchor(rmt_anchor_mat, rmt_cuboid_points)
# #cuboid_points = s3dis.transform_to_left_handed_y_up(cuboid_points)

# transformation = [0.0, 1.5, 1.5]
# rmt_cuboid_points = utils.apply_transformation_points(rmt_cuboid_points, transformation)

# rmt_cuboid_pcd = o3d.geometry.PointCloud()
# rmt_cuboid_pcd.points = o3d.utility.Vector3dVector(rmt_cuboid_points)
# #rmt_color = [184 / 255, 235 / 255, 208 / 255]
# rmt_color = [34 / 255, 137 / 255, 221 / 255]
# rmt_cuboid_pcd.colors = o3d.utility.Vector3dVector([rmt_color] * len(rmt_cuboid_pcd.points))
# rmt_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, rmt_cuboid_points)

# # Visualize
# rmt_voxel_wireframe = vis.draw_voxel_wireframe(rmt_voxel_keys, grid_size, rmt_color)

# #overlap_color = [164 / 255, 165 / 255, 174 / 255]
# overlap_color = [124 / 255, 13 / 255, 198 / 255]
# overlap_voxel_keys = set(loc_voxel_keys.keys()).intersection(set(rmt_voxel_keys.keys()))
# overlap_boxes = vis.draw_voxel_box(overlap_voxel_keys, grid_size, overlap_color)

# loc_excluded_keys = extract_excluded_keys(loc_voxel_keys, overlap_voxel_keys)
# #loc_excluded_boxes = vis.draw_voxel_box(loc_excluded_keys, grid_size, loc_color)
# loc_ex_voxel_wireframe = vis.draw_voxel_wireframe(loc_excluded_keys, grid_size, loc_color)
# rmt_excluded_keys = extract_excluded_keys(rmt_voxel_keys, overlap_voxel_keys)
# #rmt_excluded_boxes = vis.draw_voxel_box(rmt_excluded_keys, grid_size, rmt_color)
# rmt_ex_voxel_wireframe = vis.draw_voxel_wireframe(rmt_excluded_keys, grid_size, rmt_color)

# # o3d.visualization.draw_geometries(
# #         [loc_cuboid_pcd, rmt_cuboid_pcd] 
# #         # + loc_voxel_wireframe + rmt_voxel_wireframe 
# #         + overlap_boxes 
# #         + rmt_ex_voxel_wireframe + loc_ex_voxel_wireframe
# #     )

# filtered_loc_points = filter_points_by_excluded_keys(loc_cuboid_points, loc_excluded_keys, grid_size)
# filtered_rmt_points = filter_points_by_excluded_keys(rmt_cuboid_points, rmt_excluded_keys, grid_size)

# # Create new point clouds for the filtered points
# filtered_loc_pcd = o3d.geometry.PointCloud()
# filtered_loc_pcd.points = o3d.utility.Vector3dVector(filtered_loc_points)
# filtered_loc_pcd.colors = o3d.utility.Vector3dVector([loc_color] * len(filtered_loc_points))

# filtered_rmt_pcd = o3d.geometry.PointCloud()
# filtered_rmt_pcd.points = o3d.utility.Vector3dVector(filtered_rmt_points)
# filtered_rmt_pcd.colors = o3d.utility.Vector3dVector([rmt_color] * len(filtered_rmt_points))

# # Visualize filtered point clouds
# o3d.visualization.draw_geometries([filtered_loc_pcd, filtered_rmt_pcd])

# # # Generate meshes using Poisson Surface Reconstruction
# # loc_mesh = create_mesh_poisson(filtered_loc_points, depth=8)
# # rmt_mesh = create_mesh_poisson(filtered_rmt_points, depth=8)

# # # Apply colors
# # color_mesh(loc_mesh, loc_color)  # Assign loc_color to loc_mesh
# # color_mesh(rmt_mesh, rmt_color)  # Assign rmt_color to rmt_mesh

# # # Enable double-sided rendering
# # loc_mesh_double_sided = enable_double_sided_rendering(loc_mesh)
# # rmt_mesh_double_sided = enable_double_sided_rendering(rmt_mesh)

# # # Visualize the meshes
# # o3d.visualization.draw_geometries([loc_mesh_double_sided, rmt_mesh_double_sided])

# ========================================= Simple cuboid test  ========================================= 


# ============================================= S3DIS test ==============================================

# Visualize S3DIS point cloud
loc_file = "Area_1_office_1.jsonl"
loc_s3dis_points = s3dis.jsonl_to_group_clouds(loc_file)
loc_s3dis_points = utils.downsample_points(loc_s3dis_points, density_grid_size)

loc_all_points = np.vstack([
             np.hstack([points, np.full((points.shape[0], 1), const.CATEGORY_MAPPING[category_name])])
             for category_name, points in loc_s3dis_points])

loc_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, loc_all_points)
loc_voxel_wireframe = vis.draw_voxel_wireframe(loc_voxel_keys, grid_size, loc_color)

loc_s3dis_pcd = o3d.geometry.PointCloud()
loc_s3dis_pcd.points = o3d.utility.Vector3dVector(loc_all_points[:, :3])
loc_s3dis_pcd.colors = o3d.utility.Vector3dVector(s3dis.set_point_colors(loc_all_points, "rgb"))

rmt_file = "Area_5_office_39.jsonl"
rmt_s3dis_points = s3dis.jsonl_to_group_clouds(rmt_file)
rmt_s3dis_points = utils.downsample_points(rmt_s3dis_points, density_grid_size)

rmt_all_points = np.vstack([
             np.hstack([points, np.full((points.shape[0], 1), const.CATEGORY_MAPPING[category_name])])
             for category_name, points in rmt_s3dis_points])

transformation = [0.0, -2.0, 0.0]
rmt_all_points = utils.apply_transformation_points(rmt_all_points, transformation)

rmt_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, rmt_all_points)
rmt_voxel_wireframe = vis.draw_voxel_wireframe(rmt_voxel_keys, grid_size, rmt_color)

rmt_s3dis_pcd = o3d.geometry.PointCloud()
rmt_s3dis_pcd.points = o3d.utility.Vector3dVector(rmt_all_points[:, :3])
rmt_s3dis_pcd.colors = o3d.utility.Vector3dVector(s3dis.set_point_colors(rmt_all_points, "rgb"))


overlap_voxel_keys = set(loc_voxel_keys.keys()).intersection(set(rmt_voxel_keys.keys()))
overlap_boxes = vis.draw_voxel_box(overlap_voxel_keys, grid_size, overlap_color)

loc_excluded_keys = extract_excluded_keys(loc_voxel_keys, overlap_voxel_keys)
#loc_excluded_boxes = vis.draw_voxel_box(loc_excluded_keys, grid_size, loc_color)
#loc_ex_voxel_wireframe = vis.draw_voxel_wireframe(loc_excluded_keys, grid_size, loc_color)
rmt_excluded_keys = extract_excluded_keys(rmt_voxel_keys, overlap_voxel_keys)
#rmt_excluded_boxes = vis.draw_voxel_box(rmt_excluded_keys, grid_size, rmt_color)
#rmt_ex_voxel_wireframe = vis.draw_voxel_wireframe(rmt_excluded_keys, grid_size, rmt_color)

# o3d.visualization.draw_geometries(
#         [loc_cuboid_pcd, rmt_cuboid_pcd] 
#         # + loc_voxel_wireframe + rmt_voxel_wireframe 
#         + overlap_boxes 
#         + rmt_ex_voxel_wireframe + loc_ex_voxel_wireframe
#     )

filtered_loc_points = filter_points_by_excluded_keys(loc_all_points, loc_excluded_keys, grid_size)
filtered_rmt_points = filter_points_by_excluded_keys(rmt_all_points, rmt_excluded_keys, grid_size)

# Create new point clouds for the filtered points
filtered_loc_pcd = o3d.geometry.PointCloud()
filtered_loc_pcd.points = o3d.utility.Vector3dVector(filtered_loc_points[:, :3])
filtered_loc_pcd.colors = o3d.utility.Vector3dVector(s3dis.set_point_colors(filtered_loc_points, "rgb"))

filtered_rmt_pcd = o3d.geometry.PointCloud()
filtered_rmt_pcd.points = o3d.utility.Vector3dVector(filtered_rmt_points[:, :3])
filtered_rmt_pcd.colors = o3d.utility.Vector3dVector(s3dis.set_point_colors(filtered_rmt_points, "rgb"))

# Visualize filtered point clouds
o3d.visualization.draw_geometries([filtered_loc_pcd, filtered_rmt_pcd])

# o3d.visualization.draw_geometries( [loc_s3dis_pcd, rmt_s3dis_pcd] + loc_voxel_wireframe + rmt_voxel_wireframe
#         # [loc_cuboid_pcd, rmt_cuboid_pcd] 
#         # # + loc_voxel_wireframe + rmt_voxel_wireframe 
#         # + overlap_boxes 
#         # + rmt_ex_voxel_wireframe + loc_ex_voxel_wireframe
#     )

# ============================================= S3DIS test ==============================================
