import open3d as o3d
import numpy as np
import components.S3DIS_to_json as s3dis
import components.constants as const
import components.geometry_utils as utils
import components.visualization as vis
from points_to_splats import convert_points_to_splats

CUBOID_PRESETS = [
    {
        "width": 6.0,
        "height": 3.0,
        "depth": 6.0,
    },
    {
        "width": 3.0,
        "height": 3.0,
        "depth": 3.0,
    },
    {
        "width": 4.5,
        "height": 3.5,
        "depth": 3.0,
    },
    {
        "width": 4.0,
        "height": 3.0,
        "depth": 5.0,
    },
]

LOCAL_INDEX = 3
REMOTE_INDEX = 2

LOCAL_COLOR = [255 / 255, 113 / 255, 91 / 255]
REMOTE_COLOR = [34 / 255, 137 / 255, 221 / 255]
OVERLAP_COLOR = [124 / 255, 13 / 255, 198 / 255]

DENSITY = 12
GRID_SIZE = 0.12


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

def extract_bounded_keys(keys, overlapping_keys):
    # Convert overlapping_keys to a NumPy array and then to a set for efficient lookup
    overlapping_keys_set = {tuple(key) for key in overlapping_keys}
    
    # Extract actual (x,z) pairs from overlapping keys
    bound_xz = set((int(x), int(z)) for x, _, z in overlapping_keys)
    
    # Add all y within the bounds
    min_y, max_y = np.array(overlapping_keys)[:, 1].min(), np.array(overlapping_keys)[:, 1].max()
    bound_y = set(range(int(min_y), int(max_y) + 1))
    
    # Select all y values that exist in keys for (x, z)
    bounded_keys = set()
    for x, z in bound_xz:
        matching_keys = keys[(keys[:, 0] == x) & (keys[:, 2] == z)]
        all_ys = matching_keys[:, 1] if matching_keys.size > 0 else []
        for y in all_ys:
            if y in bound_y:
                key = (x, int(y), z)
                # Only add if the key is not in overlapping_keys
                if key not in overlapping_keys_set:
                    bounded_keys.add(key)
    
    return bounded_keys

def filter_points_by_excluded_keys(points, excluded_keys, grid_size):
    # Compute voxel keys for all points
    voxel_keys = np.floor(points[:, :3] / grid_size).astype(int)

    # Convert excluded_keys and voxel_keys to structured arrays for comparison
    dtype = [('x', int), ('y', int), ('z', int)]
    structured_voxel_keys = np.array([tuple(v) for v in voxel_keys], dtype=dtype)
    structured_excluded_keys = np.array([tuple(v) for v in excluded_keys], dtype=dtype)

    # Determine which points are not in the excluded keys
    mask = ~np.isin(structured_voxel_keys, structured_excluded_keys)

    # Return filtered points
    return points[mask]

# Generate the cuboid point cloud
loc_cuboid_points = create_cuboid_point_cloud(CUBOID_PRESETS[LOCAL_INDEX]["width"], 
                                              CUBOID_PRESETS[LOCAL_INDEX]["height"], 
                                              CUBOID_PRESETS[LOCAL_INDEX]["depth"],
                                              DENSITY)

loc_anchor_mat = s3dis.compute_anchor_mat(loc_cuboid_points)
loc_cuboid_points = s3dis.transform_points_by_anchor(loc_anchor_mat, loc_cuboid_points)

loc_cuboid_pcd = o3d.geometry.PointCloud()
loc_cuboid_pcd.points = o3d.utility.Vector3dVector(loc_cuboid_points)
loc_cuboid_pcd.colors = o3d.utility.Vector3dVector([LOCAL_COLOR] * len(loc_cuboid_pcd.points))
loc_voxel_keys = utils.extract_voxels_keys_points(loc_cuboid_points, GRID_SIZE)

# Visualize
rmt_cuboid_points = create_cuboid_point_cloud(CUBOID_PRESETS[REMOTE_INDEX]["width"], 
                                              CUBOID_PRESETS[REMOTE_INDEX]["height"], 
                                              CUBOID_PRESETS[REMOTE_INDEX]["depth"],
                                              DENSITY)

rmt_anchor_mat = s3dis.compute_anchor_mat(rmt_cuboid_points)
rmt_cuboid_points = s3dis.transform_points_by_anchor(rmt_anchor_mat, rmt_cuboid_points)

transformation = [60.0, -1.3, 1.0]
rmt_cuboid_points = utils.apply_transformation_points(rmt_cuboid_points, transformation)

rmt_cuboid_pcd = o3d.geometry.PointCloud()
rmt_cuboid_pcd.points = o3d.utility.Vector3dVector(rmt_cuboid_points)
rmt_cuboid_pcd.colors = o3d.utility.Vector3dVector([REMOTE_COLOR] * len(rmt_cuboid_pcd.points))
rmt_voxel_keys = utils.extract_voxels_keys_points(rmt_cuboid_points, GRID_SIZE)

# Visualize
overlap_voxel_keys = utils.extract_intersected_voxels(loc_voxel_keys, rmt_voxel_keys)
#overlap_boxes = vis.draw_voxel_box(overlap_voxel_keys, GRID_SIZE, OVERLAP_COLOR)

loc_excluded_keys = extract_bounded_keys(loc_voxel_keys, overlap_voxel_keys)
#loc_excluded_boxes = vis.draw_voxel_box(loc_excluded_keys, grid_size, loc_color)
loc_ex_voxel_wireframe = vis.draw_voxel_wireframe(loc_excluded_keys, GRID_SIZE, LOCAL_COLOR)

rmt_excluded_keys = extract_bounded_keys(rmt_voxel_keys, overlap_voxel_keys)
#rmt_excluded_boxes = vis.draw_voxel_box(rmt_excluded_keys, grid_size, rmt_color)
rmt_ex_voxel_wireframe = vis.draw_voxel_wireframe(rmt_excluded_keys, GRID_SIZE, REMOTE_COLOR)

coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
o3d.visualization.draw_geometries(
        [loc_cuboid_pcd, rmt_cuboid_pcd] 
        #+ overlap_boxes 
        + loc_ex_voxel_wireframe
        + rmt_ex_voxel_wireframe 
        + [coordinate_frame]
    )



filtered_loc_points = filter_points_by_excluded_keys(loc_cuboid_points, loc_excluded_keys, GRID_SIZE)
filtered_rmt_points = filter_points_by_excluded_keys(rmt_cuboid_points, rmt_excluded_keys, GRID_SIZE)

# Create new point clouds for the filtered points
filtered_loc_pcd = o3d.geometry.PointCloud()
filtered_loc_pcd.points = o3d.utility.Vector3dVector(filtered_loc_points)
filtered_loc_pcd.colors = o3d.utility.Vector3dVector([LOCAL_COLOR] * len(filtered_loc_points))

filtered_rmt_pcd = o3d.geometry.PointCloud()
filtered_rmt_pcd.points = o3d.utility.Vector3dVector(filtered_rmt_points)
filtered_rmt_pcd.colors = o3d.utility.Vector3dVector([REMOTE_COLOR] * len(filtered_rmt_points))

# Visualize filtered point clouds
o3d.visualization.draw_geometries([filtered_loc_pcd, filtered_rmt_pcd])


# Add colors to the points
filtered_loc_points_with_color = np.hstack([
    filtered_loc_points,
    np.tile(np.array([LOCAL_COLOR]) * 255, (len(filtered_loc_points), 1))  # Repeat color for each point
])

filtered_rmt_points_with_color = np.hstack([
    filtered_rmt_points,
    np.tile(np.array([REMOTE_COLOR]) * 255, (len(filtered_rmt_points), 1))  # Repeat color for each point
])

# Combine the points
combined_points = np.vstack([
    filtered_loc_points_with_color,
    filtered_rmt_points_with_color
])

# Convert and save as gaussian splats
output_path = "./DATA/gaussian_splats/cuboid_test.ply"
convert_points_to_splats(combined_points, output_path)
