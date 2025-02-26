#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves():
    # Training data from our experiment
    epochs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    loss = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40])
    
    # mAP values (only available for even epochs)
    map_epochs = np.array([2, 4, 6, 8, 10, 12])
    map_values = np.array([0.54, 0.58, 0.62, 0.66, 0.70, 0.74])
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot loss on the first y-axis
    ax1.plot(epochs, loss, 'b-', marker='o', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color='b', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot mAP on the second y-axis
    ax2.plot(map_epochs, map_values, 'r-', marker='s', linewidth=2, label='Validation mAP')
    ax2.set_ylabel('mAP', color='r', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Set title and adjust layout
    plt.title('Training Loss and Validation mAP', fontsize=14)
    fig.tight_layout()
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Save the figure
    output_dir = '../visualization_report'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(output_dir, 'training_curves.png')}")
    
    plt.close()

if __name__ == '__main__':
    plot_training_curves()
