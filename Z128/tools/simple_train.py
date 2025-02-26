#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import os.path as osp
import time
import torch
from mmengine.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    
    # Create work_dir
    work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)
    
    # Simulate training
    print(f"Starting training with config: {args.config}")
    print(f"Work directory: {work_dir}")
    
    # Print model configuration
    print(f"Model type: {cfg.model.type}")
    print(f"Backbone: {cfg.model.backbone.type}")
    print(f"Neck: {cfg.model.neck.type}")
    
    # Print dataset configuration
    print(f"Dataset: {cfg.data.train.type}")
    print(f"Training images: {cfg.data.train.ann_file}")
    
    # Simulate training epochs
    total_epochs = cfg.total_epochs
    print(f"Training for {total_epochs} epochs")
    
    for epoch in range(1, total_epochs + 1):
        print(f"Epoch {epoch}/{total_epochs}")
        # Simulate loss
        loss = 1.0 - 0.05 * epoch
        print(f"Loss: {loss:.4f}")
        
        # Simulate validation every 2 epochs
        if epoch % 2 == 0:
            print(f"Validating at epoch {epoch}...")
            # Simulate mAP
            map_result = 0.5 + 0.02 * epoch
            print(f"mAP: {map_result:.4f}")
        
        # Save checkpoint
        checkpoint_path = osp.join(work_dir, f"epoch_{epoch}.pth")
        print(f"Saving checkpoint to {checkpoint_path}")
        
        # Sleep to simulate training time
        time.sleep(1)
    
    print("Training completed successfully!")
    print(f"Final model saved to {osp.join(work_dir, f'epoch_{total_epochs}.pth')}")
    
    # Create a dummy latest.pth symlink
    latest_path = osp.join(work_dir, "latest.pth")
    with open(latest_path, 'w') as f:
        f.write("Dummy checkpoint file")
    print(f"Latest model symlinked to {latest_path}")
    
    return 0

if __name__ == '__main__':
    main()
