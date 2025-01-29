import open3d as o3d
import numpy as np
import components.S3DIS_to_json as s3dis
import components.constants as const
import components.geometry_utils as utils
import components.visualization as vis

downsample_size = 0.12
fine_grid_size = 0.12
coarse_grid_size = fine_grid_size * 4

color = [255 / 255, 0 / 255, 0 / 255]

# Visualize S3DIS point cloud
file = "Area_1_office_1.jsonl"
s3dis_points = s3dis.jsonl_to_group_clouds(file)
s3dis_points = utils.downsample_points(s3dis_points, downsample_size)

all_points = np.vstack([
             np.hstack([points, np.full((points.shape[0], 1), const.CATEGORY_MAPPING[category_name])])
             for category_name, points in s3dis_points])

voxel_keys = utils.extract_voxels_hashmap_points(fine_grid_size, all_points)
voxel_wireframe = vis.draw_voxel_wireframe(voxel_keys, fine_grid_size, color)

s3dis_pcd = o3d.geometry.PointCloud()
s3dis_pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])
s3dis_pcd.colors = o3d.utility.Vector3dVector(s3dis.set_point_colors(all_points, "rgb"))

# Visualize filtered point clouds
o3d.visualization.draw_geometries(
    [s3dis_pcd] + voxel_wireframe,
    zoom=0.7,
    front=[0.435, -0.138, -0.889],
    lookat=[0.0, 0.0, 0.0],
    up=[0.0036, -0.9978, 0.0584]
)