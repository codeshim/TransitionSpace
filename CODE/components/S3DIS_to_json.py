import open3d as o3d
import numpy as np
import os
import json
import argparse
from tqdm import tqdm
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

S3DIS_DIR = "DATA/S3DIS/"
JSONL_DIR = "DATA/jsonl/"

def load_point_cloud_data(directory):
    """Load annotated point cloud files grouped by their 'Annotations' directory."""
    annotation_groups = defaultdict(list)

    directory = S3DIS_DIR + directory
    print(directory)

    for root, _, files in os.walk(directory):
        if "Annotations" in root:
            annotation_groups[root].extend(
                os.path.join(root, file) for file in files if file.endswith(".txt")
            )

    for group_dir, files in annotation_groups.items():
        group_clouds = []
        folder_name = os.path.basename(os.path.dirname(group_dir))
        for file in tqdm(files, desc=f"Loading files in {group_dir}"):
            category_name = os.path.basename(file).split("_")[0]
            category_id = CATEGORY_MAPPING.get(category_name, -1)
            if category_id == -1:
                print(f"Unknown category: {category_name} in file {file}")
                continue
            data = np.loadtxt(file, dtype=float)
            group_clouds.append((category_name, data))

    return folder_name, group_clouds

# def compute_bounding_box(points):
#     """Compute the bounding box of a set of points."""
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(points[:, :3])
#     bounding_box = point_cloud.get_axis_aligned_bounding_box()
#     return np.asarray(bounding_box.get_box_points()).tolist()   # 0 - 2 - 7 - 1 | 3 - 5- 4- 6 (Clock-wise)

def compute_anchor_mat(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use X, Y, Z only
    bbox = pcd.get_axis_aligned_bounding_box()
    anchor = bbox.get_min_bound()
    mat = np.eye(4)
    mat[:3, 3] = -anchor
    return mat

def transform_points_by_anchor(anchor_mat, points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # Use X, Y, Z only
    pcd.transform(anchor_mat)
    points[:, :3] = np.asarray(pcd.points)  # Get transformed X, Y, Z
    return points

def transform_to_left_handed_y_up(points):
    """
    Right-handed y-up.
    """
    transformed_points = points.copy()
    transformed_points[:, [0, 1, 2]] = points[:, [1, 2, 0]]  # Swap y and z
    #transformed_points[:, 0] *= -1  # Invert x-axis for left-handed
    return transformed_points

def write_as_jsonl(directory):
    """Save point clouds as a JSON Lines file for extremely large datasets."""
    folder_name, group_clouds = load_point_cloud_data(directory)
    output_file = JSONL_DIR + f"{folder_name}.jsonl"

    # Combine all points with category IDs to compute anchor
    all_points = np.vstack([ points for category_name, points in group_clouds])
    #all_points = transform_to_left_handed_y_up(all_points)
    anchor_mat = compute_anchor_mat(all_points)
   
    room_data = {
        "folder_name": folder_name,
        "number_of_points": sum(data.shape[0] for _, data in group_clouds),
        "categories": []
    }
    
    with open(output_file, "w") as f:
        # Write the initial metadata line
        json.dump({
            "folder_name": room_data["folder_name"],
            "number_of_points": room_data["number_of_points"]
        }, f)
        f.write('\n')
        
        # Process each category separately
        for category_name, points in tqdm(group_clouds, desc=f"Loading files in {output_file}"):
            points = transform_points_by_anchor(anchor_mat, points)
            points = transform_to_left_handed_y_up(points)

            category_entry = {
                "category": category_name,
                "points": points.tolist()  # X, Y, Z, R, G, B
            }
            
            # Write each category as a separate JSON line
            json.dump(category_entry, f)
            f.write('\n')
            
            # Also keep track of categories in room_data (optional)            
            #room_data["categories"].append(category_entry)

    print(f"{output_file} saved.")

def jsonl_to_array(directory):
    directory = JSONL_DIR + directory
    with open(directory, 'r') as f:
        # Read the first line (metadata)
        metadata = json.loads(f.readline().strip())
        
        # Prepare lists to store point cloud data
        group_clouds = []
        
        # Read subsequent lines (category data)
        for line in tqdm(f, desc=f"Loading files in {directory}"):
            category_data = json.loads(line.strip())
            if "points" in category_data:
                points = np.array(category_data["points"])
                group_clouds.append((category_data["category"], points))

    if not group_clouds:
        print("No point cloud data loaded.")
        return
    
    return group_clouds

def set_point_colors(points, mode="rgb"):
    """Set the colors of the points based on the selected mode."""
    if points.shape[1] < 7:
        raise ValueError("Input points array must have at least 7 columns (x, y, z, r, g, b, category).")

    if mode == "rgb":
        # Extract RGB and normalize
        return points[:, 3:6] / 255.0
    
    elif mode == "category":
        # Use the last column as category index
        categories = points[:, 6].astype(int)
        colors = np.array([CATEGORY_COLORS.get(cat, [0, 0, 0]) 
                           for cat in tqdm(categories, desc=f"Setting up points color")]) / 255.0
        return colors

    else:
        raise ValueError(f"Invalid mode '{mode}'. Supported modes are 'rgb' and 'category'.")

def visualize(group_clouds):
    """Visualize point clouds from a JSONL file with RGB/Category color toggle."""
    mode = "rgb"

    def update_visualizer(vis):
        nonlocal mode
        vis.clear_geometries()
        pcd = o3d.geometry.PointCloud()
        
        # Combine all points with category IDs
        all_points = np.vstack([
                     np.hstack([points, np.full((points.shape[0], 1), CATEGORY_MAPPING[category_name])])
                     for category_name, points in group_clouds])
        
        # Set points and colors
        pcd.points = o3d.utility.Vector3dVector(all_points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(set_point_colors(all_points, mode))

        vis.add_geometry(pcd)

        # Add coordinate frame for axes (with flipped z-axis)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

    def toggle_mode(vis):
        """Toggle between RGB and category color mode."""
        nonlocal mode
        mode = "rgb" if mode == "category" else "category"
        update_visualizer(vis)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # Register key callback for toggling color mode
    vis.register_key_callback(ord('C'), toggle_mode)

    update_visualizer(vis)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    """
    Example
    python CODE/S3DIS_to_json.py -w --folder Area_1_office_1
    python CODE/S3DIS_to_json.py -r --jsonl Area_1_office_1.jsonl
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-w", action="store_true", help="Write mode: Save the point cloud data as jsonl.")
    parser.add_argument("-r", action="store_true", help="Read mode: Visualize point cloud data from a jsonl file.")
    parser.add_argument("--folder", type=str, help="Path to the folder containing point cloud data (required in write mode).")
    parser.add_argument("--jsonl", type=str, help="Path to a jsonl file to read and visualize (required in read mode).")

    args = parser.parse_args()

    if args.w:
        if not args.folder:
            print("Error: --folder is required in write mode (-w).")
        else:
            write_as_jsonl(args.folder)

    elif args.r:
        if not args.jsonl:
            print("Error: --jsonl is required in read mode (-r).")
        else:
            group_clouds = jsonl_to_array(args.jsonl)
            visualize(group_clouds)
    else:
        print("Error: Either -w or -r must be specified.")
