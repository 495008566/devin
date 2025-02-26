#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import json
from shapely.geometry import Polygon

def convert_dota_to_points(data_root, split='train'):
    """Convert DOTA OBB annotations to point annotations (center points)."""
    img_dir = os.path.join(data_root, split, 'images')
    label_dir = os.path.join(data_root, split, 'labelTxt')
    output_file = os.path.join(data_root, split, 'point_annotations.json')
    
    point_annotations = {}
    
    # Get all label files
    label_files = os.listdir(label_dir)
    
    for label_file in label_files:
        if not label_file.endswith('.txt'):
            continue
            
        img_name = label_file.replace('.txt', '.png')
        if not os.path.exists(os.path.join(img_dir, img_name)):
            img_name = label_file.replace('.txt', '.jpg')
            if not os.path.exists(os.path.join(img_dir, img_name)):
                continue
        
        # Initialize entry for this image
        point_annotations[img_name] = {
            'file_name': img_name,
            'points': [],
            'labels': []
        }
        
        # Read label file
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            
        # Skip header line if it exists
        if lines and 'imagesource' in lines[0]:
            lines = lines[2:]
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
                
            # Extract coordinates and class
            coords = [float(x) for x in parts[:8]]
            class_name = parts[8]
            
            # Convert to polygon and get centroid
            poly = Polygon([(coords[i], coords[i+1]) for i in range(0, 8, 2)])
            centroid = poly.centroid
            
            # Add point annotation
            point_annotations[img_name]['points'].append([centroid.x, centroid.y])
            point_annotations[img_name]['labels'].append(class_name)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(point_annotations, f)
        
    print(f"Converted {len(point_annotations)} images to point annotations.")
    return point_annotations

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert DOTA OBB annotations to point annotations')
    parser.add_argument('--data-root', type=str, default='data/dota', help='Path to DOTA dataset')
    parser.add_argument('--splits', type=str, default='train,val', help='Dataset splits to convert')
    args = parser.parse_args()
    
    for split in args.splits.split(','):
        convert_dota_to_points(args.data_root, split)
