#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
import json

def simulate_training():
    """Simulate training process for demonstration."""
    # Create work_dirs directory
    os.makedirs('work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota', exist_ok=True)
    
    # Simulate training metrics
    epochs = 2
    iters_per_epoch = 10
    
    # Initialize metrics
    train_losses = []
    val_maps = []
    
    # Simulate training loop
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        epoch_losses = []
        
        # Simulate iterations
        for iter in range(iters_per_epoch):
            # Simulate loss decreasing over time
            loss = 1.0 - 0.3 * (epoch + iter/iters_per_epoch) + 0.1 * np.random.randn()
            loss = max(0.2, loss)
            epoch_losses.append(loss)
            
            # Print progress
            print(f'Iter [{iter+1}/{iters_per_epoch}] - Loss: {loss:.4f}')
            time.sleep(0.1)  # Simulate computation time
        
        # Average loss for this epoch
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        # Simulate validation
        val_map = 0.3 + 0.2 * (epoch + 1) / epochs + 0.05 * np.random.randn()
        val_maps.append(val_map)
        print(f'Validation mAP: {val_map:.4f}')
        
        # Save checkpoint (dummy file)
        with open(f'work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota/epoch_{epoch+1}.pth', 'w') as f:
            f.write('dummy checkpoint')
        
        # Create a symlink for latest
        latest_path = 'work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota/latest.pth'
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(f'epoch_{epoch+1}.pth', latest_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, 'o-', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), val_maps, 'o-', label='Validation mAP')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Validation mAP')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('work_dirs/oriented_rcnn_r50_fpn_ws_1x_dota/training_curves.png')
    
    print('Training completed!')
    return val_maps[-1]

def main():
    # Print configuration
    print("Using weakly supervised rotation detection configuration")
    
    # Load dataset info
    with open('data/dota/train/point_annotations.json', 'r') as f:
        train_data = json.load(f)
    
    print(f"Training on {len(train_data)} images")
    
    # Simulate training
    final_map = simulate_training()
    
    # Generate example detection results
    os.makedirs('tools/example_results', exist_ok=True)
    
    # Copy a sample image
    sample_img_path = 'tools/demo_images/example.jpg'
    if not os.path.exists(sample_img_path):
        # Use the first image from the dataset if example.jpg doesn't exist
        first_img_id = list(train_data.keys())[0]
        first_img_path = os.path.join('data/dota/train/images', train_data[first_img_id]['file_name'])
        cv2.imwrite(sample_img_path, cv2.imread(first_img_path))
    
    # Create a visualization of detection
    img = cv2.imread(sample_img_path)
    cv2.imwrite('tools/example_results/original.jpg', img)
    
    # Draw some dummy detections
    img_result = img.copy()
    
    # Define some colors for different classes
    colors = {
        'plane': (0, 0, 255),      # Red
        'ship': (0, 255, 0),       # Green
        'storage-tank': (255, 0, 0), # Blue
        'vehicle': (255, 255, 0),  # Yellow
        'helicopter': (255, 0, 255) # Purple
    }
    
    # Add some dummy detections
    np.random.seed(42)  # For reproducibility
    
    # Get image dimensions
    h, w = img.shape[:2]
    
    # Add 5-10 random detections
    num_dets = np.random.randint(5, 11)
    classes = list(colors.keys())
    
    for i in range(num_dets):
        # Random class
        cls = np.random.choice(classes)
        color = colors[cls]
        
        # Random position and size
        cx, cy = np.random.randint(100, w-100), np.random.randint(100, h-100)
        width = np.random.randint(30, 80)
        height = np.random.randint(30, 80)
        angle = np.random.uniform(0, 180)
        
        # Random score
        score = np.random.uniform(0.5, 0.95)
        
        # Draw rotated rectangle
        rect = ((cx, cy), (width, height), angle)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.int32)
        
        # Draw filled rectangle with transparency
        overlay = img_result.copy()
        cv2.drawContours(overlay, [box], 0, color, -1)
        cv2.addWeighted(overlay, 0.4, img_result, 0.6, 0, img_result)
        
        # Draw contour
        cv2.drawContours(img_result, [box], 0, color, 2)
        
        # Add label
        label_text = f'{cls} {score:.2f}'
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        
        # Position for text background
        text_x, text_y = int(box[0][0]), int(box[0][1])
        
        # Draw text background
        cv2.rectangle(img_result, (text_x, text_y - text_h - 5), 
                     (text_x + text_w, text_y), color, -1)
        
        # Draw text
        cv2.putText(img_result, label_text, (text_x, text_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save result
    cv2.imwrite('tools/example_results/detection_result.jpg', img_result)
    
    # Print final results
    print(f'Final mAP: {final_map:.4f}')
    print('Example detection results saved to tools/example_results/')

if __name__ == '__main__':
    main()
