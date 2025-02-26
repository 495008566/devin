#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import argparse
import glob
import json
import numpy as np
import cv2
from tqdm import tqdm
from shapely.geometry import Polygon


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare DOTA dataset for weakly supervised learning')
    parser.add_argument('--data-root', help='DOTA dataset root')
    parser.add_argument('--output-dir', help='Output directory')
    args = parser.parse_args()
    return args


def parse_dota_annotation(ann_file):
    """Parse DOTA annotation file to get oriented bounding boxes."""
    with open(ann_file, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # Extract coordinates and class name
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
        class_name = parts[8]
        
        # Create annotation
        annotation = {
            'points': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            'class_name': class_name
        }
        annotations.append(annotation)
    
    return annotations


def convert_to_point_annotations(annotations):
    """Convert oriented bounding box annotations to point annotations."""
    point_annotations = []
    for ann in annotations:
        points = np.array(ann['points'])
        # Calculate center point
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        point_annotation = {
            'point': [center_x, center_y],
            'class_name': ann['class_name']
        }
        point_annotations.append(point_annotation)
    
    return point_annotations


def process_dota_dataset(data_root, output_dir, split='train'):
    """Process DOTA dataset for weakly supervised learning."""
    # Create output directories
    os.makedirs(osp.join(output_dir, split, 'images'), exist_ok=True)
    
    # Get image and annotation files
    img_dir = osp.join(data_root, 'images')
    ann_dir = osp.join(data_root, 'labelTxt')
    
    img_files = glob.glob(osp.join(img_dir, '*.png'))
    
    # Process each image
    point_annotations_dict = {}
    for img_file in tqdm(img_files, desc=f'Processing {split} set'):
        img_name = osp.basename(img_file)
        img_id = osp.splitext(img_name)[0]
        
        # Read image
        img = cv2.imread(img_file)
        if img is None:
            print(f"Warning: Could not read image {img_file}")
            continue
        
        # Get annotation file
        ann_file = osp.join(ann_dir, f'{img_id}.txt')
        if not osp.exists(ann_file):
            print(f"Warning: Annotation file {ann_file} does not exist")
            continue
        
        # Parse annotations
        annotations = parse_dota_annotation(ann_file)
        point_annotations = convert_to_point_annotations(annotations)
        
        # Save image
        output_img_path = osp.join(output_dir, split, 'images', img_name)
        cv2.imwrite(output_img_path, img)
        
        # Store point annotations
        point_annotations_dict[img_id] = {
            'file_name': img_name,
            'width': img.shape[1],
            'height': img.shape[0],
            'points': point_annotations
        }
    
    # Save point annotations to JSON file
    with open(osp.join(output_dir, split, 'point_annotations.json'), 'w') as f:
        json.dump(point_annotations_dict, f)


def main():
    args = parse_args()
    
    # Process train, val, and test sets
    for split in ['train', 'val', 'test']:
        process_dota_dataset(
            osp.join(args.data_root, split),
            args.output_dir,
            split
        )
    
    print('Done!')


if __name__ == '__main__':
    main()
