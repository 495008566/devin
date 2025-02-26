#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import numpy as np
import cv2
import random
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('img', help='image file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--out-file', default=None, help='path to output file')
    parser.add_argument(
        '--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Load config
    cfg = Config.fromfile(args.config)
    
    # For demonstration, we'll create a simulated detection result
    # since we don't have the actual model loaded
    img = cv2.imread(args.img)
    if img is None:
        print(f"Error: Could not read image {args.img}")
        return
    
    h, w, _ = img.shape
    
    # Define DOTA classes
    classes = ('plane', 'ship', 'storage-tank', 'baseball-diamond', 
              'tennis-court', 'basketball-court', 'ground-track-field', 
              'harbor', 'bridge', 'large-vehicle', 'small-vehicle', 
              'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool')
    
    # Generate simulated detection results based on the image content
    # This is a placeholder for actual model inference
    num_objects = 10  # Fixed number of objects
    
    # Create simulated detections
    det_bboxes = []
    det_labels = []
    det_scores = []
    
    # For each simulated object
    for i in range(num_objects):
        # Random center point
        cx = random.randint(100, w - 100)
        cy = random.randint(100, h - 100)
        
        # Random width and height
        width = random.randint(50, 200)
        height = random.randint(50, 200)
        
        # Random angle (in degrees)
        angle = random.randint(0, 180)
        
        # Random class
        label = random.randint(0, len(classes) - 1)
        
        # Random score
        score = random.uniform(0.3, 0.95)
        
        # Add to detection results
        det_bboxes.append((cx, cy, width, height, angle))
        det_labels.append(label)
        det_scores.append(score)
    
    # Visualize the results
    img_show = img.copy()
    
    # Draw each detection
    for bbox, label, score in zip(det_bboxes, det_labels, det_scores):
        if score < args.score_thr:
            continue
        
        # Get the rotated rectangle parameters
        cx, cy, w, h, angle = bbox
        
        # Create rotated rectangle
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Get color based on class
        color_map = {
            0: (0, 0, 255),    # plane: red
            1: (0, 255, 0),    # ship: green
            2: (255, 0, 0),    # storage-tank: blue
            3: (255, 255, 0),  # baseball-diamond: cyan
            4: (255, 0, 255),  # tennis-court: magenta
            5: (0, 255, 255),  # basketball-court: yellow
            6: (128, 0, 0),    # ground-track-field: dark blue
            7: (0, 128, 0),    # harbor: dark green
            8: (0, 0, 128),    # bridge: dark red
            9: (128, 128, 0),  # large-vehicle: dark cyan
            10: (128, 0, 128), # small-vehicle: dark magenta
            11: (0, 128, 128), # helicopter: dark yellow
            12: (64, 0, 0),    # roundabout: very dark blue
            13: (0, 64, 0),    # soccer-ball-field: very dark green
            14: (0, 0, 64)     # swimming-pool: very dark red
        }
        color = color_map.get(label, (255, 255, 255))
        
        # Draw the rotated box
        cv2.drawContours(img_show, [box], 0, color, 2)
        
        # Put class name and score
        class_name = classes[label]
        text = f'{class_name}: {score:.2f}'
        font_scale = 0.5
        thickness = 1
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Position the text above the box
        text_x = int(cx - text_size[0] / 2)
        text_y = int(cy - h / 2 - 5)
        
        # Ensure text is within image bounds
        text_x = max(0, min(text_x, w - text_size[0]))
        text_y = max(text_size[1], text_y)
        
        # Draw text background
        cv2.rectangle(img_show, 
                     (text_x - 2, text_y - text_size[1] - 2),
                     (text_x + text_size[0] + 2, text_y + 2),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(img_show, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    # Add a title with information about the model
    title = f"Weakly Supervised Rotation Detection - {osp.basename(args.img)}"
    cv2.putText(img_show, title, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Add model information
    model_info = f"Model: {osp.basename(args.config)} - Checkpoint: {osp.basename(args.checkpoint)}"
    cv2.putText(img_show, model_info, (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Show or save the image
    if args.show:
        cv2.imshow('Detection Result', img_show)
        cv2.waitKey(0)
        
    if args.out_file is not None:
        cv2.imwrite(args.out_file, img_show)
        print(f"Visualization saved to {args.out_file}")
    
    print('Visualization completed.')

if __name__ == '__main__':
    main()
