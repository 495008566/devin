#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import cv2
import json
import random
from tqdm import tqdm

def generate_synthetic_image(img_size=1024, num_objects=10):
    """Generate a synthetic image with random objects."""
    # Create a blank image
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
    
    # Add some background texture
    for _ in range(20):
        x1, y1 = random.randint(0, img_size-100), random.randint(0, img_size-100)
        x2, y2 = x1 + random.randint(50, 100), y1 + random.randint(50, 100)
        color = (random.randint(200, 235), random.randint(200, 235), random.randint(200, 235))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    # List to store object annotations
    annotations = []
    
    # Add random objects
    classes = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court', 
               'basketball-court', 'ground-track-field', 'harbor', 'bridge', 
               'large-vehicle', 'small-vehicle', 'helicopter', 'roundabout', 
               'soccer-ball-field', 'swimming-pool']
    
    for _ in range(num_objects):
        # Random class
        class_name = random.choice(classes)
        
        # Random position and size
        cx, cy = random.randint(100, img_size-100), random.randint(100, img_size-100)
        w = random.randint(30, 80)
        h = random.randint(30, 80)
        angle = random.uniform(0, 180)  # Rotation angle in degrees
        
        # Calculate rotated rectangle corners
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Draw the object
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        cv2.drawContours(img, [box], 0, color, -1)
        
        # Add border
        cv2.drawContours(img, [box], 0, (0, 0, 0), 2)
        
        # Store annotation
        x1, y1 = box[0]
        x2, y2 = box[1]
        x3, y3 = box[2]
        x4, y4 = box[3]
        
        annotation = {
            'points': [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
            'class_name': class_name
        }
        annotations.append(annotation)
    
    return img, annotations

def generate_dota_dataset(output_dir, num_images=10, split='train'):
    """Generate a synthetic DOTA dataset."""
    # Create directories
    img_dir = osp.join(output_dir, split, 'images')
    label_dir = osp.join(output_dir, split, 'labelTxt')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Generate images and annotations
    point_annotations_dict = {}
    
    for i in tqdm(range(num_images), desc=f'Generating {split} set'):
        img_id = f'P{i:04d}'
        img_name = f'{img_id}.png'
        
        # Generate image and annotations
        img, annotations = generate_synthetic_image()
        
        # Save image
        img_path = osp.join(img_dir, img_name)
        cv2.imwrite(img_path, img)
        
        # Save DOTA format annotation (txt)
        ann_path = osp.join(label_dir, f'{img_id}.txt')
        with open(ann_path, 'w') as f:
            f.write('imagesource:Synthetic\ngsd:1.0\n')
            for ann in annotations:
                points = ann['points']
                class_name = ann['class_name']
                line = ' '.join([f'{p[0]} {p[1]}' for p in points])
                f.write(f'{line} {class_name} 0\n')
        
        # Create point annotations for weakly supervised learning
        point_annotations = []
        for ann in annotations:
            points = np.array(ann['points'])
            # Calculate center point
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            point_annotation = {
                'point': [float(center_x), float(center_y)],
                'class_name': ann['class_name']
            }
            point_annotations.append(point_annotation)
        
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
    
    print(f'Generated {num_images} images for {split} set')

def main():
    # Generate datasets for train, val, and test
    output_dir = 'data/dota'
    generate_dota_dataset(output_dir, num_images=5, split='train')
    generate_dota_dataset(output_dir, num_images=2, split='val')
    generate_dota_dataset(output_dir, num_images=2, split='test')
    
    print('Done!')

if __name__ == '__main__':
    main()
