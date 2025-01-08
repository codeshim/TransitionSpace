import argparse
from components.S3DIS_to_json import jsonl_to_group_clouds
import components.geometry_utils as utils
import components.constants as const
from components.optimization import strength_pareto_evolutionary_algorithm_2
from components.visualization import visualize_and_record_pareto_front
import open3d as o3d


"""
Example:
python CODE/main.py --loc Area_1_office_1.jsonl --rmt Area_5_office_39.jsonl --grid_size 0.2 --generation 1 --down_size 0.009
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local room JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote room JSONL file.")
    parser.add_argument("--grid_size", type=float, default=0.5)
    parser.add_argument("--generation", type=int, default=50)
    parser.add_argument("--down_size", type=float, default=0.05)
    
    args = parser.parse_args()

    # Load point clouds
    print("\nLoading clouds...")
    const.g_loc_name = args.loc
    const.g_rmt_name = args.rmt
    const.g_local_cloud = jsonl_to_group_clouds(const.g_loc_name)  
    const.g_remote_cloud = jsonl_to_group_clouds(const.g_rmt_name)

    # Downsample point clouds
    print("\nDownsampling clouds...")
    const.g_down_size = args.down_size
    const.g_local_cloud = utils.downsample_points(const.g_local_cloud)
    const.g_remote_cloud = utils.downsample_points(const.g_remote_cloud)

    # Initialize global values(constants)
    const.g_grid_size = args.grid_size
    const.param_generations = args.generation
    const.g_remote_centroid = utils.get_cloud_centroid(const.g_remote_cloud)

    # Polygon
    #const.g_local_polygon = utils.extract_free_space_polygon(const.g_local_cloud)

    # Voxel
    print("\nExtract voxels...")
    const.g_loc_strt_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_structure_categories)
    const.g_loc_feat_voxels = utils.extract_selected_voxels_keys(const.g_local_cloud, const.g_feature_categories)
    rmt_strt_voxels = utils.extract_selected_voxels_keys(const.g_remote_cloud, const.g_structure_categories)
    rmt_feat_voxels = utils.extract_selected_voxels_keys(const.g_remote_cloud, const.g_feature_categories)
    const.g_obj1_min = 0.0
    const.g_obj1_max = min(len(const.g_loc_strt_voxels), len(rmt_strt_voxels))
    const.g_obj2_min = 0.0
    const.g_obj2_max = min(len(const.g_loc_feat_voxels), len(rmt_feat_voxels))
    

    # strength_pareto_evolutionary_algorithm_2 will be placed here***
    pareto_front = strength_pareto_evolutionary_algorithm_2(
                    population_size=const.param_population_size,
                    archive_size=const.param_archive_size,
                    mutation_rate=const.param_mutation_rate,
                    min_values=const.DEFAULT_MIN_VALUES,
                    max_values=const.DEFAULT_MAX_VALUES,
                    generations=const.param_generations,
                    verbose=False,)

    # Add cleanup code
    import gc
    gc.collect()
    
    const.g_best_tr = pareto_front[:3]
    const.g_best_obj1 = pareto_front[3]
    const.g_best_obj1 = pareto_front[4]
    
    # visualize pareto_front[0]
    visualize_and_record_pareto_front(record=False)
    # temp = [60.0, 2.5, 1.0]
    # visualize_pareto_front(temp)
