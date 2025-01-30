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

density_grid_size = 0.01
grid_size = 0.2

loc_color = [255 / 255, 113 / 255, 91 / 255]
rmt_color = [34 / 255, 137 / 255, 221 / 255]
overlap_color = [124 / 255, 13 / 255, 198 / 255]

# Parameters for the cuboid
loc_width = 6.0
loc_depth = 6.0
loc_height = 3.0
loc_density = 10

# Generate the cuboid point cloud
loc_cuboid_points = create_cuboid_point_cloud(loc_width, loc_height, loc_depth, loc_density)

loc_anchor_mat = s3dis.compute_anchor_mat(loc_cuboid_points)
loc_cuboid_points = s3dis.transform_points_by_anchor(loc_anchor_mat, loc_cuboid_points)
#cuboid_points = s3dis.transform_to_left_handed_y_up(cuboid_points)

loc_cuboid_pcd = o3d.geometry.PointCloud()
loc_cuboid_pcd.points = o3d.utility.Vector3dVector(loc_cuboid_points)
#loc_color = [141 / 255, 90 / 255, 151 / 255]
loc_color = [255 / 255, 113 / 255, 91 / 255]
loc_cuboid_pcd.colors = o3d.utility.Vector3dVector([loc_color] * len(loc_cuboid_pcd.points))
loc_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, loc_cuboid_points)

# Visualize
loc_voxel_wireframe = vis.draw_voxel_wireframe(loc_voxel_keys, grid_size, loc_color)

# Parameters for the cuboid
rmt_width = 3.0
rmt_depth = 3.0
rmt_height = 3.0
rmt_density = 10

rmt_cuboid_points = create_cuboid_point_cloud(rmt_width, rmt_height, rmt_depth, rmt_density)

rmt_anchor_mat = s3dis.compute_anchor_mat(rmt_cuboid_points)
rmt_cuboid_points = s3dis.transform_points_by_anchor(rmt_anchor_mat, rmt_cuboid_points)
#cuboid_points = s3dis.transform_to_left_handed_y_up(cuboid_points)

transformation = [0.0, 1.5, 1.5]
rmt_cuboid_points = utils.apply_transformation_points(rmt_cuboid_points, transformation)

rmt_cuboid_pcd = o3d.geometry.PointCloud()
rmt_cuboid_pcd.points = o3d.utility.Vector3dVector(rmt_cuboid_points)
#rmt_color = [184 / 255, 235 / 255, 208 / 255]
rmt_color = [34 / 255, 137 / 255, 221 / 255]
rmt_cuboid_pcd.colors = o3d.utility.Vector3dVector([rmt_color] * len(rmt_cuboid_pcd.points))
rmt_voxel_keys = utils.extract_voxels_hashmap_points(grid_size, rmt_cuboid_points)

# Visualize
rmt_voxel_wireframe = vis.draw_voxel_wireframe(rmt_voxel_keys, grid_size, rmt_color)

#overlap_color = [164 / 255, 165 / 255, 174 / 255]
overlap_color = [124 / 255, 13 / 255, 198 / 255]
overlap_voxel_keys = set(loc_voxel_keys.keys()).intersection(set(rmt_voxel_keys.keys()))
overlap_boxes = vis.draw_voxel_box(overlap_voxel_keys, grid_size, overlap_color)

loc_excluded_keys = utils.extract_excluded_keys(loc_voxel_keys, overlap_voxel_keys)
#loc_excluded_boxes = vis.draw_voxel_box(loc_excluded_keys, grid_size, loc_color)
loc_ex_voxel_wireframe = vis.draw_voxel_wireframe(loc_excluded_keys, grid_size, loc_color)
rmt_excluded_keys = utils.extract_excluded_keys(rmt_voxel_keys, overlap_voxel_keys)
#rmt_excluded_boxes = vis.draw_voxel_box(rmt_excluded_keys, grid_size, rmt_color)
rmt_ex_voxel_wireframe = vis.draw_voxel_wireframe(rmt_excluded_keys, grid_size, rmt_color)

# o3d.visualization.draw_geometries(
#         [loc_cuboid_pcd, rmt_cuboid_pcd] 
#         # + loc_voxel_wireframe + rmt_voxel_wireframe 
#         + overlap_boxes 
#         + rmt_ex_voxel_wireframe + loc_ex_voxel_wireframe
#     )

filtered_loc_points = filter_points_by_excluded_keys(loc_cuboid_points, loc_excluded_keys, grid_size)
filtered_rmt_points = filter_points_by_excluded_keys(rmt_cuboid_points, rmt_excluded_keys, grid_size)

# Create new point clouds for the filtered points
filtered_loc_pcd = o3d.geometry.PointCloud()
filtered_loc_pcd.points = o3d.utility.Vector3dVector(filtered_loc_points)
filtered_loc_pcd.colors = o3d.utility.Vector3dVector([loc_color] * len(filtered_loc_points))

filtered_rmt_pcd = o3d.geometry.PointCloud()
filtered_rmt_pcd.points = o3d.utility.Vector3dVector(filtered_rmt_points)
filtered_rmt_pcd.colors = o3d.utility.Vector3dVector([rmt_color] * len(filtered_rmt_points))

# Visualize filtered point clouds
o3d.visualization.draw_geometries([filtered_loc_pcd, filtered_rmt_pcd])
