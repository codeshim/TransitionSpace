import os
from datetime import datetime
import matplotlib.pyplot as plt
import components.constants as const

OUT_DIR = "OUT/"

def formatted_start_time():
    formatted_start_time = datetime.fromtimestamp(const.g_total_start_time).strftime("%Y-%m-%d_%H-%M-%S")
    return formatted_start_time

def capture_screenshot(vis):
    """Capture and save a screenshot of the current visualization."""
    base_filename = os.path.join(OUT_DIR, f"{formatted_start_time()}_3d")
    count = 0

    # Loop until a unique filename is found
    while True:
        filename = f"{base_filename}_{count}.jpg"
        if not os.path.exists(filename):
            break  # Exit the loop if the filename does not exist
        count += 1

    # Save the screenshot
    vis.capture_screen_image(filename)
    print(f"3D screenshot saved as: {filename}")

def save_plt_fig(title):
    filename = os.path.join(OUT_DIR, f"{formatted_start_time()}_{title}.jpg")
    plt.savefig(filename)
    print(f"plt screenshot saved as: {filename}")

def save_report():
    """
    Save a report for the SPEA2 optimization process.
    """
    # Ensure the output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Format start time for filename and report
    filename = f"{formatted_start_time()}_report.txt"
    filepath = os.path.join(OUT_DIR, filename)

    # Create the report content
    report_content = f"""
    Optimization Report
    ==========================
    Start Time: {formatted_start_time()}

    Datas:
    --------------------------
    Local Space: {const.g_loc_name}
    Remote Space: {const.g_rmt_name}

    Results:
    --------------------------
    Transformation: {const.g_best_tr}
    Objective 1: {const.g_best_obj1} 
    Objective 2: {const.g_best_obj2}

    Parameters:
    --------------------------
    Downsample Size: {const.g_down_size}
    Grid Size: {const.g_grid_size}
    Population Size: {const.param_population_size}
    Archive Size: {const.param_archive_size}
    Mutation Rate: {const.param_mutation_rate}
    Min Values: {const.DEFAULT_MIN_VALUES}
    Max Values: {const.DEFAULT_MAX_VALUES}
    Generations: {const.param_generations}   

    Timing:
    --------------------------
    Total Elapsed Time: {const.g_total_elapsed_time:.2f} seconds
    Average Time per Generation: {const.g_average_generation_time:.2f} seconds

    Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """

    # Write the report to the file
    with open(filepath, "w") as report_file:
        report_file.write(report_content.strip())

    print(f"Report saved to: {filepath}")