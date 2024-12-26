import argparse
from components.S3DIS_to_json import jsonl_to_group_clouds
import components.geometry_utils as utils
import components.constants as const
from components.optimization import strength_pareto_evolutionary_algorithm_2
from components.visualization import visualize_pareto_front


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
    const.g_local_cloud = jsonl_to_group_clouds(args.loc)  
    const.g_remote_cloud = jsonl_to_group_clouds(args.rmt)

    # Downsample point clouds
    const.g_down_size = args.down_size
    const.g_local_cloud = utils.downsample_points(const.g_local_cloud)
    const.g_remote_cloud = utils.downsample_points(const.g_remote_cloud)

    # Initialize global values(constants)
    const.g_grid_size = args.grid_size
    const.param_generations = args.generation
    const.g_local_polygon = utils.extract_free_space_polygon(const.g_local_cloud)
    const.g_local_voxels = utils.extract_voxels_hashmap(const.g_local_cloud)
    const.g_remote_centroid = utils.get_cloud_centroid(const.g_remote_cloud)

    # strength_pareto_evolutionary_algorithm_2 will be placed here***
    pareto_front = strength_pareto_evolutionary_algorithm_2(
                    population_size=const.param_population_size,
                    archive_size=const.param_archive_size,
                    mutation_rate=const.param_mutation_rate,
                    min_values=const.DEFAULT_MIN_VALUES,
                    max_values=const.DEFAULT_MAX_VALUES,
                    generations=const.param_generations,
                    verbose=True,)
    
    # visualize pareto_front[0]
    visualize_pareto_front(pareto_front[0])
    # temp = [60.0, 2.5, 1.0]
    # visualize_pareto_front(temp)
