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

# Default global values for point cloud and optimization operations
g_local_cloud = []
g_local_polygon = None
g_local_voxels = None
g_remote_cloud = []
g_remote_centroid = None
g_down_size = 0.05
g_grid_size = 0.5  # Default grid size for voxelization
g_included_category = "ceiling"
g_excluded_categories = [
    "wall", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"
]

# Storage for optimization-related data
g_voxel_loops = defaultdict(list)
g_shared_polygon = defaultdict(list)

# Bounds for optimization variables
DEFAULT_MIN_VALUES = [-180.0, -5.0, -5.0]
DEFAULT_MAX_VALUES = [180.0, 5.0, 5.0]

# SPEA2 parameters
param_population_size = 20
param_archive_size = 20
param_mutation_rate = 0.1
param_generations = 50