from collections import defaultdict

# Define category code colors
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

CATEGORY_COLORS = {
    0: [92, 164, 169],
    1: [219, 84, 97],
    2: [255, 217, 206],
    3: [89, 60, 143],
    4: [142, 249, 243],
    5: [23, 23, 56],
    6: [37, 110, 255],
    7: [70, 35, 122],
    8: [61, 220, 151],
    9: [255, 73, 92],
    10: [237, 106, 90],
    11: [244, 241, 187],
    12: [255, 0, 0]
}

# Data info
g_loc_name = ""
g_rmt_name = ""

# Default global values for point cloud and optimization operations
g_local_cloud = []
g_local_polygon = None
#g_local_voxels = None
g_loc_strt_voxels = None
g_loc_feat_voxels = None
g_remote_cloud = []
g_remote_centroid = None
g_down_size = 0.05
g_grid_size = 0.2  # Default grid size for voxelization

# For polygon
g_included_category = "ceiling"
g_excluded_categories = [
    "wall", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
]

# For voxel hashmap
g_structure_categories = ["ceiling", "floor", "wall", "beam", "column"]
g_feature_categories = ["window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


# Storage for optimization-related data
#g_voxel_loops = defaultdict(list)
g_shared_polygon = defaultdict(list)    
g_overlap_strt_voxels = defaultdict(list)
g_overlap_feat_voxels = defaultdict(list)

# Bounds for optimization variables
DEFAULT_MIN_VALUES = [-180.0, -6.0, -6.0]
DEFAULT_MAX_VALUES = [180.0, 6.0, 6.0]

# SPEA2 parameters
param_population_size = 50
param_archive_size = 50
param_mutation_rate = 0.2
param_generations = 50

# Bounds
g_obj1_min = 0.0
g_obj1_max = 0.0
g_obj2_min = 0.0
g_obj2_max = 0.0

# Results
g_best_tr = []
g_best_obj1 = 0.0
g_best_obj2 = 0.0
g_best_obj1_list = []
g_best_obj2_list = []

# Time
g_total_start_time = 0.0
g_total_elapsed_time = 0.0
g_average_generation_time = 0.0