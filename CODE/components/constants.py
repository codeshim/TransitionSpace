from collections import defaultdict

# Default global values for point cloud and optimization operations
g_local_cloud = []
g_local_polygon = None
g_local_voxels = None
g_remote_cloud = []
g_remote_centroid = None
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