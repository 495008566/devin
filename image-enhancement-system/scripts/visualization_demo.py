#!/usr/bin/env python3
"""
Visualization Demo for Image Enhancement System.
This script demonstrates the visual aspects of the image enhancement system,
showing before/after comparisons for various enhancement techniques.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.processor import ImageProcessor
from src.utils.comparison import create_side_by_side_comparison, plot_histograms

def main():
    """Main function to run the visualization demo."""
    print("Starting Image Enhancement System Visualization Demo...")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Define directories
    input_dir = os.path.join("data", "input", "samples")
    output_dir = os.path.join("data", "output", "demo")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # List of demo images and techniques
    demos = [
        {
            "title": "Grayscale Conversion and Contrast Adjustment",
            "image": "peppers.tiff",
            "techniques": [
                ("Original", lambda img: img),
                ("Grayscale", processor.convert_to_grayscale),
                ("Contrast Enhanced", lambda img: processor.adjust_contrast(img, alpha=1.5, beta=10)),
                ("Histogram Equalized", processor.histogram_equalization)
            ]
        },
        {
            "title": "Noise Reduction",
            "image": "lena.tiff",
            "techniques": [
                ("Original", lambda img: img),
                ("Mean Filter", lambda img: processor.apply_mean_filter(img, kernel_size=5)),
                ("Gaussian Filter", lambda img: processor.apply_gaussian_filter(img, kernel_size=5, sigma=1.5)),
                ("Median Filter", lambda img: processor.apply_median_filter(img, kernel_size=5))
            ]
        },
        {
            "title": "Edge Enhancement and Sharpening",
            "image": "house.tiff",
            "techniques": [
                ("Original", lambda img: img),
                ("Sobel Edges", lambda img: processor.detect_edges_sobel(processor.convert_to_grayscale(img))),
                ("Sharpened", lambda img: processor.sharpen_image(img, amount=1.5)),
                ("Laplacian Sharpened", processor.laplacian_sharpening)
            ]
        },
        {
            "title": "Image Filtering",
            "image": "mandrill.tiff",
            "techniques": [
                ("Original", lambda img: img),
                ("Bilateral Filter", lambda img: processor.apply_bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)),
                ("Emboss Filter", lambda img: processor.apply_custom_filter(img, np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]))),
                ("Edge Filter", lambda img: processor.apply_custom_filter(img, np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])))
            ]
        },
        {
            "title": "Interpolation and Magnification",
            "image": "resolution_chart.tiff",
            "techniques": [
                ("Original", lambda img: img),
                ("Nearest Neighbor", lambda img: processor.resize_image(img, scale_factor=2.0, interpolation=cv2.INTER_NEAREST)),
                ("Bilinear", lambda img: processor.resize_image(img, scale_factor=2.0, interpolation=cv2.INTER_LINEAR)),
                ("Bicubic", lambda img: processor.resize_image(img, scale_factor=2.0, interpolation=cv2.INTER_CUBIC))
            ]
        },
        {
            "title": "Pseudocolor Processing",
            "image": "mri.tiff",
            "techniques": [
                ("Original", lambda img: processor.convert_to_grayscale(img) if len(img.shape) == 3 else img),
                ("Jet Colormap", lambda img: processor.apply_pseudocolor(processor.convert_to_grayscale(img) if len(img.shape) == 3 else img, colormap=cv2.COLORMAP_JET)),
                ("Hot Colormap", lambda img: processor.apply_pseudocolor(processor.convert_to_grayscale(img) if len(img.shape) == 3 else img, colormap=cv2.COLORMAP_HOT)),
                ("False Color", lambda img: processor.apply_false_color(processor.convert_to_grayscale(img) if len(img.shape) == 3 else img))
            ]
        }
    ]
    
    # Process each demo
    for i, demo in enumerate(demos):
        print(f"\nProcessing demo {i+1}/{len(demos)}: {demo['title']}")
        
        # Load image
        image_path = os.path.join(input_dir, demo["image"])
        original = processor.load_image(image_path)
        
        if original is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Apply techniques
        results = []
        labels = []
        
        for label, technique in demo["techniques"]:
            try:
                result = technique(original)
                results.append(result)
                labels.append(label)
            except Exception as e:
                print(f"Error applying {label}: {str(e)}")
        
        # Create grid visualization
        fig, axes = plt.subplots(1, len(results), figsize=(16, 5))
        fig.suptitle(demo["title"], fontsize=16)
        
        for j, (result, label) in enumerate(zip(results, labels)):
            if len(result.shape) == 2:
                axes[j].imshow(result, cmap='gray')
            else:
                # Convert BGR to RGB for matplotlib
                if result.shape[2] == 3:
                    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                    axes[j].imshow(result_rgb)
                else:
                    axes[j].imshow(result)
            
            axes[j].set_title(label)
            axes[j].axis('off')
        
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(output_dir, f"demo_{i+1}_{demo['title'].lower().replace(' ', '_')}.png")
        plt.savefig(output_path, dpi=150)
        print(f"Saved visualization to {output_path}")
        
        # Create interactive slider comparison for the first and last technique
        if len(results) >= 2:
            first = results[0]
            last = results[-1]
            
            # Convert to RGB for matplotlib
            if len(first.shape) == 3 and first.shape[2] == 3:
                first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
            if len(last.shape) == 3 and last.shape[2] == 3:
                last = cv2.cvtColor(last, cv2.COLOR_BGR2RGB)
            
            # Create figure for slider comparison
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.subplots_adjust(bottom=0.25)
            
            # Display the first image
            if len(first.shape) == 2:
                ax_img = ax.imshow(first, cmap='gray')
            else:
                ax_img = ax.imshow(first)
            
            ax.set_title(f"Interactive Comparison: {labels[0]} vs {labels[-1]}")
            ax.axis('off')
            
            # Create slider
            from matplotlib.widgets import Slider
            slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
            slider = Slider(slider_ax, 'Blend', 0, 1, valinit=0)
            
            # Update function for slider
            def update(val):
                # Blend the two images based on slider value
                if len(first.shape) == 2 and len(last.shape) == 2:
                    blended = (1 - val) * first + val * last
                    ax_img.set_data(blended)
                elif len(first.shape) == 3 and len(last.shape) == 3:
                    blended = (1 - val) * first + val * last
                    ax_img.set_data(blended.astype(np.uint8))
                fig.canvas.draw_idle()
            
            slider.on_changed(update)
            
            # Save interactive comparison
            output_path = os.path.join(output_dir, f"interactive_{i+1}_{demo['title'].lower().replace(' ', '_')}.png")
            plt.savefig(output_path, dpi=150)
            print(f"Saved interactive comparison to {output_path}")
    
    print("\nVisualization demo completed. All results saved to:", os.path.abspath(output_dir))

if __name__ == "__main__":
    main()
