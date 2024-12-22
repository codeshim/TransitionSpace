import open3d as o3d
import numpy as np
import os
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

def load_point_cloud_data(directory):
    """Load annotated point cloud files grouped by their 'Annotations' directory."""
    annotation_groups = defaultdict(list)  # Dictionary to group files by directory
    overall_clouds = []

    # Collect annotation files grouped by their parent directory
    for root, dirs, files in os.walk(directory):
        if "Annotations" in root:  # Check for 'Annotations' in the path
            annotation_groups[root].extend(
                os.path.join(root, file) for file in files if file.endswith(".txt")
            )

    # Process annotation files group by group
    for group_dir, files in annotation_groups.items():
        print(f"Processing directory: {group_dir}")
        group_clouds = []  # List to hold point clouds for this group
        folder_name = os.path.basename(os.path.dirname(group_dir))  # Folder above "Annotations"
        for file in tqdm(files, desc=f"Loading files in {group_dir}"):
            try:
                # Extract category name from the filename
                category_name = os.path.basename(file).split("_")[0]
                category_id = CATEGORY_MAPPING.get(category_name, -1)
                if category_id == -1:
                    print(f"Unknown category: {category_name} in file {file}")
                    continue  # Skip unknown categories

                data = np.loadtxt(file, dtype=float)  # Load X Y Z points
                category_column = np.full((data.shape[0], 1), category_id)  # Add category column
                group_clouds.append(np.hstack((data, category_column)))  # Add to group
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        # Combine all clouds in this group and add to overall clouds
        if group_clouds:
            combined_cloud = np.vstack(group_clouds)
            overall_clouds.append((folder_name, combined_cloud))  # Add folder name and cloud

    return overall_clouds  # List of tuples: [(folder_name, point_cloud), ...]

def set_point_colors(points, mode="rgb"):
    """Set the colors of the points based on the selected mode."""
    if mode == "rgb":
        return points[:, 3:6] / 255.0
    elif mode == "category":
        colors = np.zeros((len(points), 3))
        for i, point in enumerate(points):
            category = int(point[6])  # Category is the last column
            colors[i] = np.array(CATEGORY_COLORS.get(category, [0, 0, 0])) / 255.0
        return colors

def visualize_point_clouds(directory):
    """Visualize point clouds with Open3D and save screenshots using the <P> key."""
    point_cloud_data = load_point_cloud_data(directory)
    if not point_cloud_data:
        print("No point cloud data loaded.")
        return

    current_index = 0
    mode = "rgb"

    def update_visualizer(vis):
        nonlocal mode, current_index
        vis.clear_geometries()
        if len(point_cloud_data) == 0:
            print("No point cloud data to display.")
            return
        folder_name, points = point_cloud_data[current_index]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(set_point_colors(points, mode))
        vis.add_geometry(pcd)
        print(f"Currently displaying: {current_index + 1}/{len(point_cloud_data)} - Folder: {folder_name}")

    def toggle_mode(vis):
        """Toggle between RGB and category color mode."""
        nonlocal mode
        mode = "rgb" if mode == "category" else "category"
        update_visualizer(vis)

    def next_cloud(vis):
        """Go to the next point cloud."""
        nonlocal current_index
        current_index = (current_index + 1) % len(point_cloud_data)
        update_visualizer(vis)

    def prev_cloud(vis):
        """Go to the previous point cloud."""
        nonlocal current_index
        current_index = (current_index - 1) % len(point_cloud_data)
        update_visualizer(vis)

    def capture_screenshot(vis):
        """Capture and save a screenshot of the current visualization."""
        folder_name, _ = point_cloud_data[current_index]
        filename = f"{folder_name}_screenshot.png"
        vis.capture_screen_image(filename)
        print(f"Screenshot saved as: {filename}")

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.register_key_callback(ord('C'), toggle_mode)
    vis.register_key_callback(262, next_cloud)  # Right arrow key
    vis.register_key_callback(263, prev_cloud)  # Left arrow key
    vis.register_key_callback(ord('P'), capture_screenshot)  # 'P' key to take a screenshot

    update_visualizer(vis)
    vis.run()
    vis.destroy_window()

# Directory containing the annotated point cloud data
point_cloud_directory = "DATA/S3DIS/"
visualize_point_clouds(point_cloud_directory)