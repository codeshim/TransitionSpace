import open3d as o3d
import numpy as np
from components.S3DIS_to_json import jsonl_to_group_clouds
import components.geometry_utils as utils
import components.constants as const
import components.visualization as v
import components.objective_functions as f


"""
Example:
python CODE/main.py --loc Area_3_lounge_1.jsonl --rmt Area_3_office_8.jsonl --grid_size 0.12 --generation 2^C--down_size 0.12 --ismultiobj
"""

if __name__ == "__main__":
    # Load point clouds
    print("\nLoading clouds...")
    const.g_loc_name = "Area_3_lounge_1.jsonl"
    const.g_rmt_name = "Area_3_office_8.jsonl"
    const.g_local_cloud_origin = jsonl_to_group_clouds(const.g_loc_name)  
    const.g_remote_cloud_origin = jsonl_to_group_clouds(const.g_rmt_name)

    const.g_remote_centroid = utils.get_cloud_centroid(const.g_remote_cloud_origin)
    transformation = tuple([75.57716547, -1.12732314, 0.94579129])
    transformed_remote_cloud_origin = utils.apply_points_transformation(const.g_remote_cloud_origin, const.g_remote_centroid, transformation)


    # Downsample point clouds
    print("\nDownsampling clouds...")
    const.g_local_cloud = utils.downsample_points(const.g_local_cloud_origin)
    transformed_remote_cloud = utils.downsample_points(transformed_remote_cloud_origin)
    

    # Voxel
    print("\nExtract voxels...")
    # # ============================= Maximize overlapped structure voxels =============================
    const.g_loc_strt_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_structure_categories)
    rmt_strt_voxels = utils.extract_selected_voxels_keys(transformed_remote_cloud, const.g_structure_categories)
    const.g_loc_feat_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_feature_categories)    
    rmt_feat_voxels = utils.extract_selected_voxels_keys(transformed_remote_cloud, const.g_feature_categories)
    
    # Filter out floor voxels that overlap with features within height threshold
    height_percentage = 0.4 # Exclude overlaps within bottom 10% of height
    loc_strt_filtered = utils.filter_floor_voxels(const.g_loc_strt_voxels, const.g_loc_feat_voxels, height_percentage)
    rmt_strt_filtered = utils.filter_floor_voxels(rmt_strt_voxels, rmt_feat_voxels, height_percentage)

    overlapping_strt_voxels = utils.extract_intersected_voxels(loc_strt_filtered, rmt_strt_filtered)
    overlapping_feat_voxels = utils.extract_intersected_voxels(const.g_loc_feat_voxels, rmt_feat_voxels)

    # Visualization
    vis_sets = []
    #vis_sets = v.visualize_wireframe_voxels(const.g_loc_strt_voxels)
    vis_sets = v.visualize_filled_voxels(overlapping_strt_voxels)

    local_all_cloud_points = np.vstack([points for _, points in const.g_local_cloud_origin])
    local_cloud_o3d = o3d.geometry.PointCloud()
    local_cloud_o3d.points = o3d.utility.Vector3dVector(local_all_cloud_points[:, :3])
    local_cloud_o3d.colors = o3d.utility.Vector3dVector(local_all_cloud_points[:, 3:6] / 255.0)

    remote_all_cloud_points = np.vstack([points for _, points in transformed_remote_cloud_origin])
    remote_cloud_o3d = o3d.geometry.PointCloud()
    remote_cloud_o3d.points = o3d.utility.Vector3dVector(remote_all_cloud_points[:, :3])
    remote_cloud_o3d.colors = o3d.utility.Vector3dVector(remote_all_cloud_points[:, 3:6] / 255.0)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(local_cloud_o3d)
    vis.add_geometry(remote_cloud_o3d)
    for vis_item in vis_sets:
        vis.add_geometry(vis_item)
    vis.run()
    vis.destroy_window()
