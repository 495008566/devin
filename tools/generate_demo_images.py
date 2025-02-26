#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import random

def generate_synthetic_image(width=1024, height=1024, num_objects=20, output_path='demo_image.jpg'):
    """Generate a synthetic remote sensing image with objects."""
    # Create a background
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Add some texture to the background (simulating terrain)
    for _ in range(100):
        x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
        x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
        color = (random.randint(200, 240), random.randint(200, 240), random.randint(200, 240))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness=random.randint(1, 3))
    
    # Add some larger background features
    for _ in range(5):
        x, y = random.randint(0, width-1), random.randint(0, height-1)
        radius = random.randint(50, 200)
        color = (random.randint(180, 220), random.randint(180, 220), random.randint(180, 220))
        cv2.circle(img, (x, y), radius, color, -1)
    
    # Add objects
    for _ in range(num_objects):
        # Random object type
        obj_type = random.choice(['plane', 'ship', 'storage-tank', 'vehicle'])
        
        # Random position
        cx, cy = random.randint(100, width-100), random.randint(100, height-100)
        
        if obj_type == 'plane':
            # Draw a plane-like shape
            length = random.randint(40, 80)
            width = random.randint(30, 50)
            angle = random.randint(0, 360)
            color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
            
            # Create a rotated rectangle
            rect = ((cx, cy), (length, width), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, color, -1)
            
            # Add wings
            wing_length = random.randint(20, 40)
            wing_width = random.randint(5, 15)
            wing_angle = angle + 90
            wing_rect = ((cx, cy), (wing_length, wing_width), wing_angle)
            wing_box = cv2.boxPoints(wing_rect)
            wing_box = np.int0(wing_box)
            cv2.drawContours(img, [wing_box], 0, color, -1)
            
        elif obj_type == 'ship':
            # Draw a ship-like shape
            length = random.randint(50, 100)
            width = random.randint(15, 30)
            angle = random.randint(0, 360)
            color = (random.randint(50, 100), random.randint(50, 100), random.randint(100, 150))
            
            # Create a rotated rectangle
            rect = ((cx, cy), (length, width), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, color, -1)
            
        elif obj_type == 'storage-tank':
            # Draw a circular storage tank
            radius = random.randint(15, 40)
            color = (random.randint(100, 150), random.randint(100, 150), random.randint(100, 150))
            cv2.circle(img, (cx, cy), radius, color, -1)
            
            # Add a shadow
            shadow_offset = random.randint(2, 5)
            shadow_color = (color[0]-30, color[1]-30, color[2]-30)
            cv2.circle(img, (cx+shadow_offset, cy+shadow_offset), radius, shadow_color, 2)
            
        elif obj_type == 'vehicle':
            # Draw a vehicle-like shape
            length = random.randint(10, 30)
            width = random.randint(5, 15)
            angle = random.randint(0, 360)
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            
            # Create a rotated rectangle
            rect = ((cx, cy), (length, width), angle)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, color, -1)
    
    # Add some roads
    for _ in range(3):
        x1, y1 = random.randint(0, width-1), random.randint(0, height-1)
        x2, y2 = random.randint(0, width-1), random.randint(0, height-1)
        road_width = random.randint(5, 15)
        cv2.line(img, (x1, y1), (x2, y2), (100, 100, 100), thickness=road_width)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Generated synthetic image saved to {output_path}")
    return img

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs("demo_images", exist_ok=True)
    
    # Generate three different synthetic images
    generate_synthetic_image(width=1024, height=1024, num_objects=15, output_path="demo_images/synthetic_dota1.jpg")
    generate_synthetic_image(width=1024, height=1024, num_objects=20, output_path="demo_images/synthetic_dota2.jpg")
    generate_synthetic_image(width=1024, height=1024, num_objects=25, output_path="demo_images/synthetic_dota3.jpg")
    
    print("Generated 3 synthetic images in demo_images/ directory")
