import numpy as np
from plyfile import PlyElement, PlyData
import os
import json
from tqdm import tqdm
import argparse
import open3d as o3d
from components.geometry_utils import downsample_points

def estimate_normals(xyz, k_neighbors=30):
    """Estimate normals using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(k_neighbors)
    )
    
    # Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k_neighbors)
    
    return np.asarray(pcd.normals)

def normal_to_rotation(normals):
    """Convert normal vectors to rotation quaternions.
    This will align the local y-axis with the normal direction,
    making the splat perpendicular to the normal."""
    
    # Initialize quaternions array
    rotations = np.zeros((len(normals), 4), dtype=np.float32)
    
    for i, normal in enumerate(normals):
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # Find rotation between [0,1,0] and normal (using y-axis)
        v1 = np.array([0, 1, 0])
        v2 = normal
        
        # Handle special cases
        if np.allclose(v1, v2):
            # Vectors are parallel, no rotation needed
            rotations[i] = [1, 0, 0, 0]
        elif np.allclose(v1, -v2):
            # Vectors are antiparallel, rotate 180 degrees around x-axis
            rotations[i] = [0, 1, 0, 0]
        else:
            # General case
            cross_prod = np.cross(v1, v2)
            dot_prod = np.dot(v1, v2)
            
            w = np.sqrt((1.0 + dot_prod) * 2.0)
            if w < 1e-6:
                rotations[i] = [0, 1, 0, 0]  # 180-degree rotation around x-axis
            else:
                cross_prod = cross_prod / w
                rotations[i] = np.array([
                    w / 2.0,
                    cross_prod[0],
                    cross_prod[1],
                    cross_prod[2]
                ])
        
        # Add small random rotation around normal axis for variety
        angle = np.random.uniform(0, 2 * np.pi)
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        normal_rot = np.array([c, s * normal[0], s * normal[1], s * normal[2]])
        
        # Combine rotations using quaternion multiplication
        rotations[i] = quaternion_multiply(rotations[i], normal_rot)
        
        # Normalize quaternion
        rotations[i] = rotations[i] / np.linalg.norm(rotations[i])
    
    return rotations

def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def generate_diverse_scales(num_points, base_scale_mean=-3.0):
    """Generate diverse scales for Gaussian splats with subtle variations.
    The splats will be flat disks perpendicular to the normal."""
    # Generate random base scales
    base_scales = np.random.normal(base_scale_mean, 0.2, (num_points, 3)).astype(np.float32)
    
    # Add subtle variety to the scales while maintaining disk shape
    scales = base_scales.copy()
    
    # Make splats wider in x and z directions (perpendicular to normal/y-axis)
    disk_scale = np.random.uniform(0.9, 2.0, (num_points, 1))
    scales[:, [0, 2]] += np.log(disk_scale)  # Add in log space since scales are in log space
    
    # Make splats very thin in y direction (along normal)
    scales[:, 1] -= 1.5  # Make them thinner along normal direction
    
    # Add subtle random variation to make some splats slightly elliptical
    elongation_mask = np.random.random(num_points) < 0.2
    elongation_factor = np.random.uniform(0.1, 0.2, num_points)
    scales[elongation_mask, 0] += elongation_factor[elongation_mask]
    scales[elongation_mask, 2] -= elongation_factor[elongation_mask]
    
    return scales

def points_to_gaussian_data(points, random_seed=42):
    """Convert points to gaussian splat format with randomized parameters."""
    np.random.seed(random_seed)
    
    # Extract xyz and rgb from points
    xyz = points[:, :3].astype(np.float32)
    rgb = points[:, 3:6].astype(np.float32) / 255.0  # Normalize RGB to [0,1]
    
    num_points = len(points)
    
    # Estimate normals and convert to rotations
    print("Estimating normals...")
    normals = estimate_normals(xyz)
    rotations = normal_to_rotation(normals)
    
    # Generate diverse scales
    scales = generate_diverse_scales(num_points)
    
    # Opacity (logit space, will be sigmoid'd during loading)
    # Reduce opacity variation
    opacities = np.random.normal(2, 0.2, (num_points, 1)).astype(np.float32)  # Reduced standard deviation
    # Make fewer splats transparent and reduce transparency
    transparent_mask = np.random.random(num_points) < 0.1  # Reduce probability
    opacities[transparent_mask] -= 1.0  # Reduced transparency
    
    # Convert RGB to spherical harmonics DC component
    features_dc = (rgb - 0.5) / 0.28209
    
    # Random spherical harmonics coefficients for higher degrees
    max_sh_degree = 3
    num_sh_coeffs = (max_sh_degree + 1) ** 2 - 1
    # Further reduce the magnitude of higher-order SH coefficients
    features_extra = np.random.normal(0, 0.002, (num_points, num_sh_coeffs * 3)).astype(np.float32)
    
    return xyz, rotations, scales, opacities, features_dc, features_extra


def save_as_ply(output_path, xyz, rotations, scales, opacities, features_dc, features_extra):
    """Save the gaussian data as a PLY file."""
    
    # Prepare data for PLY format
    vertex_data = []
    
    # Basic properties
    vertex_data.extend([
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4')
    ])
    
    # Rotation quaternions
    for i in range(4):
        vertex_data.append((f'rot_{i}', 'f4'))
    
    # Scale parameters
    for i in range(3):
        vertex_data.append((f'scale_{i}', 'f4'))
    
    # Opacity
    vertex_data.append(('opacity', 'f4'))
    
    # SH coefficients - DC term
    for i in range(3):
        vertex_data.append((f'f_dc_{i}', 'f4'))
    
    # SH coefficients - Higher order terms
    for i in range(features_extra.shape[1]):
        vertex_data.append((f'f_rest_{i}', 'f4'))
    
    # Create structured array
    vertices = np.empty(len(xyz), dtype=vertex_data)
    
    # Fill data
    vertices['x'], vertices['y'], vertices['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    for i in range(4):
        vertices[f'rot_{i}'] = rotations[:, i]
    for i in range(3):
        vertices[f'scale_{i}'] = scales[:, i]
    vertices['opacity'] = opacities[:, 0]
    for i in range(3):
        vertices[f'f_dc_{i}'] = features_dc[:, i]
    for i in range(features_extra.shape[1]):
        vertices[f'f_rest_{i}'] = features_extra[:, i]
    
    # Create PLY element and save
    vertex_element = PlyElement.describe(vertices, 'vertex')
    PlyData([vertex_element]).write(output_path)

def downsample_points_direct(points, voxel_size):
    """Downsample points directly using Open3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255.0)
    
    # Downsample using voxel grid
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    
    # Convert back to numpy array
    xyz = np.asarray(downsampled_pcd.points)
    rgb = np.asarray(downsampled_pcd.colors) * 255.0
    
    return np.hstack([xyz, rgb])

def convert_jsonl_to_splats(jsonl_path, output_path, voxel_size):
    """Convert S3DIS JSONL data to Gaussian splat PLY format."""
    
    # Read JSONL file
    all_points = []
    with open(jsonl_path, 'r') as f:
        # Skip metadata line
        f.readline()
        # Read point data
        for line in tqdm(f, desc="Reading JSONL"):
            data = json.loads(line)
            if "points" in data:
                points = np.array(data["points"])
                all_points.append(points)
    
    # Combine all points
    combined_points = np.vstack(all_points)
    
    # Downsample points
    print(f"Original points: {len(combined_points)}")
    downsampled_points = downsample_points_direct(combined_points, voxel_size)
    print(f"Downsampled points: {len(downsampled_points)}")
    
    # Convert to gaussian format
    xyz, rotations, scales, opacities, features_dc, features_extra = points_to_gaussian_data(downsampled_points)
    
    # Save as PLY
    save_as_ply(output_path, xyz, rotations, scales, opacities, features_dc, features_extra)
    print(f"Saved converted data to {output_path}")

def convert_points_to_splats(points, output_path, random_seed=42):
    # Convert to gaussian format
    xyz, rotations, scales, opacities, features_dc, features_extra = points_to_gaussian_data(points, random_seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PLY
    save_as_ply(output_path, xyz, rotations, scales, opacities, features_dc, features_extra)
    print(f"Saved converted data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert S3DIS point cloud to Gaussian Splat format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output PLY file path")
    parser.add_argument("--k-neighbors", type=int, default=30, 
                        help="Number of neighbors for normal estimation (default: 30)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="Voxel size for point cloud downsampling (default: 0.05)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    convert_jsonl_to_splats(args.input, args.output, args.voxel_size)