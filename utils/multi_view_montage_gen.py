#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-View Monrage Generator for Orthodontic Assessment.
Part of the official implementation for:
'Foundation-Model-Guided Multi-View Multiple Instance Learning for Automated Orthodontic Assessment'

"""

import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Clinical image standards for the grid
BASE_WIDTH = 2048
BASE_HEIGHT = 1200

def pad_and_resize(image, size=(BASE_WIDTH, BASE_HEIGHT)):
    """Resizes image maintaining aspect ratio and pads to reach target size."""
    h, w = image.shape[:2]
    scale = min(size[0] / w, size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    pad_top = (size[1] - new_h) // 2
    pad_bottom = size[1] - new_h - pad_top
    pad_left = (size[0] - new_w) // 2
    pad_right = size[0] - new_w - pad_left
    
    img = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                             cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return img

def process_images(input_folder, output_folder):
    """Groups views by patient key and creates a 2x3 grid layout."""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    images_dict = {}

    # Gather files and group by patient/session key
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_extensions)]
    
    if not files:
        print(f"No valid images found in {input_folder}")
        return

    for file in files:
        fname = os.path.splitext(file)[0]
        parts = fname.split('_')
        
        # Everything before the last underscore is the patient/case key
        key = "_".join(parts[:-1]).upper()
        view = parts[-1].lower() # e.g., 'right', 'center', 'up'
        
        if key not in images_dict:
            images_dict[key] = {}
        images_dict[key][view] = os.path.join(input_folder, file)
    
    print(f"Found {len(images_dict)} unique patient keys. Generating grids...")

    for key, paths in images_dict.items():
        # Create a blank canvas: 2 rows (Height*2) x 3 columns (Width*3)
        grid = np.zeros((BASE_HEIGHT * 2, BASE_WIDTH * 3, 3), dtype=np.uint8)
        
        # Map anatomical views to specific grid coordinates
        # Row 0: Right, Center, Left | Row 1: Up, Down, (Empty)
        positions = {
            "right": (0, 0), 
            "center": (0, BASE_WIDTH), 
            "left": (0, BASE_WIDTH * 2), 
            "up": (BASE_HEIGHT, 0), 
            "down": (BASE_HEIGHT, BASE_WIDTH)
        }
        
        for view, pos in positions.items():
            if view in paths:
                img = cv2.imread(paths[view], cv2.IMREAD_COLOR)
                if img is not None:
                    img = pad_and_resize(img)
                    grid[pos[0]:pos[0] + BASE_HEIGHT, pos[1]:pos[1] + BASE_WIDTH] = img
        
        output_path = os.path.join(output_folder, f"{key}.png")
        cv2.imwrite(output_path, grid)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate clinical montages for orthodontic assessment.")
    parser.add_argument("--input", type=str, required=True, help="Path to intra_oral_images folder")
    parser.add_argument("--output", type=str, required=True, help="Path to save the montage_layout folder")
    
    args = parser.parse_args()
    
    process_images(args.input, args.output)
    print("Done.")