import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from shapely.geometry import MultiPoint
from CODE.components.S3DIS_to_json import jsonl_to_array, visualize

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

class room:
    def __init__(self):
        # group_clouds (category_name, points)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc", type=str, required=True, help="Path to the local JSONL file.")
    parser.add_argument("--rmt", type=str, required=True, help="Path to the remote JSONL file.")
    parser.add_argument("--grid_size", type=float, default=0.2)   

    args = parser.parse_args()

    local_clouds = jsonl_to_array(args.loc)
    remote_clouds = jsonl_to_array(args.rmt)

    grid_size = args.grid_size

    # visualize(local_clouds)
    # visualize(remote_clouds)


