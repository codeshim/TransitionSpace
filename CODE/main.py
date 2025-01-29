# ------------ main.py ------------ 
import argparse
import open3d as o3d
from components.S3DIS_to_json import jsonl_to_group_clouds
import components.geometry_utils as utils
import components.constants as const
from components.optimization import strength_pareto_evolutionary_algorithm_2
from components.visualization import visualize_and_record_pareto_front
import components.objective_functions as f


"""
Example:
python CODE/main.py --loc Area_3_lounge_1.jsonl --rmt Area_3_office_8.jsonl --grid_size 0.12 --generation 2^C--down_size 0.12 --ismultiobj
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local room JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote room JSONL file.")
    parser.add_argument("--grid_size", type=float, default=0.5)
    parser.add_argument("--generation", type=int, default=50)
    parser.add_argument("--down_size", type=float, default=0.05)
    parser.add_argument("--ismultiobj", action="store_true")
    parser.add_argument("--isallvoxel", action="store_true")
    
    args = parser.parse_args()

    const.g_ismulitobj = args.ismultiobj
    const.g_isallvoxel = args.isallvoxel

    # Load point clouds
    print("\nLoading clouds...")
    const.g_loc_name = args.loc
    const.g_rmt_name = args.rmt
    const.g_local_cloud_origin = jsonl_to_group_clouds(const.g_loc_name)  
    const.g_remote_cloud_origin = jsonl_to_group_clouds(const.g_rmt_name)

    # Downsample point clouds
    print("\nDownsampling clouds...")
    const.g_down_size = args.down_size
    const.g_local_cloud = utils.downsample_points(const.g_local_cloud_origin)
    const.g_remote_cloud = utils.downsample_points(const.g_remote_cloud_origin)
    const.g_remote_centroid = utils.get_cloud_centroid(const.g_remote_cloud)
    const.g_remote_strt_points = utils.extract_selected_points(const.g_remote_cloud, const.g_structure_categories)
    const.g_remote_feat_points = utils.extract_selected_points(const.g_remote_cloud, const.g_feature_categories)

    # Initialize global values(constants)
    const.g_grid_size = args.grid_size
    const.param_generations = args.generation

    if (not const.g_isallvoxel):
        # Polygon
        # ============================ Maximize overlapped structure polygons ============================
        const.g_local_polygon = utils.extract_free_space_polygon(const.g_local_cloud)        
        remote_polygon = utils.extract_free_space_polygon(const.g_remote_cloud)
        const.g_obj1_min = 0.0
        const.g_obj1_max = min(const.g_local_polygon.area, remote_polygon.area)
        # ============================ Maximize overlapped structure polygons ============================
    else:
        # Voxel
        # ============================= Maximize overlapped structure voxels =============================
        const.g_loc_strt_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_structure_categories)
        remote_strt_voxels = utils.extract_voxels_keys_points(const.g_remote_strt_points)
        const.g_obj1_min = 0.0
        const.g_obj1_max = min(len(const.g_loc_strt_voxels), len(remote_strt_voxels))
        # ============================= Maximize overlapped structure voxels =============================


    const.g_loc_feat_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_feature_categories)    
    remote_feat_voxels = utils.extract_voxels_keys_points(const.g_remote_feat_points)
    const.g_obj2_min = 0.0
    const.g_obj2_max = min(len(const.g_loc_feat_voxels), len(remote_feat_voxels))

    # strength_pareto_evolutionary_algorithm_2 will be placed here***
    pareto_front = strength_pareto_evolutionary_algorithm_2(
                    population_size=const.param_population_size,
                    archive_size=const.param_archive_size,
                    mutation_rate=const.param_mutation_rate,
                    min_values=const.DEFAULT_MIN_VALUES,
                    max_values=const.DEFAULT_MAX_VALUES,
                    generations=const.param_generations,
                    verbose=False,)

    const.g_best_tr = pareto_front[:3]
    const.g_best_obj1 = pareto_front[3]
    
    if (const.g_ismulitobj):
        const.g_best_obj2 = pareto_front[4]
    else: const.g_best_obj2 = f.obj2_transitionspace(const.g_best_tr[0], const.g_best_tr[1], const.g_best_tr[2])
    
    # visualize pareto_front[0]
    visualize_and_record_pareto_front(record=True)
    # temp = [60.0, 2.5, 1.0]
    # visualize_pareto_front(temp)
