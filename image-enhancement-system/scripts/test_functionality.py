#!/usr/bin/env python3
"""
Test script for the Image Enhancement System core functionality.
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
from src.core.metrics import get_image_metrics
from src.utils.comparison import create_side_by_side_comparison, plot_histograms
from src.utils.helpers import create_directory_if_not_exists

# Define directories
INPUT_DIR = os.path.join("data", "input", "samples")
OUTPUT_DIR = os.path.join("data", "output", "tests")

# Create output directory
create_directory_if_not_exists(OUTPUT_DIR)

def test_grayscale_conversion():
    """Test grayscale conversion functionality."""
    print("\n=== Testing Grayscale Conversion ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load a color image
    image_path = os.path.join(INPUT_DIR, "peppers.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Apply grayscale conversion
    grayscale = processor.convert_to_grayscale(original)
    
    # Save result
    output_path = os.path.join(OUTPUT_DIR, "grayscale_peppers.jpg")
    processor.save_image(grayscale, output_path)
    
    # Create comparison
    comparison = create_side_by_side_comparison(original, grayscale, ("Original", "Grayscale"))
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_grayscale.jpg")
    processor.save_image(comparison, comparison_path)
    
    print(f"Grayscale conversion test completed. Results saved to {output_path}")
    print(f"Comparison saved to {comparison_path}")
    
    return True

def test_contrast_adjustment():
    """Test contrast adjustment functionality."""
    print("\n=== Testing Contrast Adjustment ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load a low contrast image
    image_path = os.path.join(INPUT_DIR, "moon.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Apply contrast adjustment
    enhanced = processor.adjust_contrast(original, alpha=1.8, beta=10)
    
    # Save result
    output_path = os.path.join(OUTPUT_DIR, "contrast_enhanced_moon.jpg")
    processor.save_image(enhanced, output_path)
    
    # Create comparison
    comparison = create_side_by_side_comparison(original, enhanced, ("Original", "Contrast Enhanced"))
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_contrast.jpg")
    processor.save_image(comparison, comparison_path)
    
    # Test histogram equalization
    hist_eq = processor.histogram_equalization(original)
    hist_eq_path = os.path.join(OUTPUT_DIR, "histogram_eq_moon.jpg")
    processor.save_image(hist_eq, hist_eq_path)
    
    # Create histogram comparison
    hist_comparison = create_side_by_side_comparison(original, hist_eq, ("Original", "Histogram Equalized"))
    hist_comparison_path = os.path.join(OUTPUT_DIR, "comparison_hist_eq.jpg")
    processor.save_image(hist_comparison, hist_comparison_path)
    
    print(f"Contrast adjustment test completed. Results saved to {output_path}")
    print(f"Histogram equalization test completed. Results saved to {hist_eq_path}")
    print(f"Comparisons saved to {comparison_path} and {hist_comparison_path}")
    
    return True

def test_noise_reduction():
    """Test noise reduction using mean filter."""
    print("\n=== Testing Noise Reduction ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load an image
    image_path = os.path.join(INPUT_DIR, "lena.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Add some noise to the image
    noisy = original.copy()
    noise = np.random.normal(0, 25, original.shape).astype(np.uint8)
    noisy = cv2.add(original, noise)
    
    # Save noisy image
    noisy_path = os.path.join(OUTPUT_DIR, "noisy_lena.jpg")
    processor.save_image(noisy, noisy_path)
    
    # Apply mean filter
    mean_filtered = processor.apply_mean_filter(noisy, kernel_size=5)
    mean_path = os.path.join(OUTPUT_DIR, "mean_filtered_lena.jpg")
    processor.save_image(mean_filtered, mean_path)
    
    # Apply Gaussian filter
    gaussian_filtered = processor.apply_gaussian_filter(noisy, kernel_size=5, sigma=1.5)
    gaussian_path = os.path.join(OUTPUT_DIR, "gaussian_filtered_lena.jpg")
    processor.save_image(gaussian_filtered, gaussian_path)
    
    # Apply median filter
    median_filtered = processor.apply_median_filter(noisy, kernel_size=5)
    median_path = os.path.join(OUTPUT_DIR, "median_filtered_lena.jpg")
    processor.save_image(median_filtered, median_path)
    
    # Create comparison
    # Combine all images into a grid
    top_row = np.hstack((original, noisy))
    bottom_row = np.hstack((mean_filtered, median_filtered))
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)
    
    # Create label backgrounds
    h, w = original.shape[:2]
    label_height = 30
    
    label_bg1 = np.zeros((label_height, w, 3), dtype=np.uint8)
    label_bg2 = np.zeros((label_height, w, 3), dtype=np.uint8)
    label_bg3 = np.zeros((label_height, w, 3), dtype=np.uint8)
    label_bg4 = np.zeros((label_height, w, 3), dtype=np.uint8)
    
    cv2.putText(label_bg1, "Original", (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(label_bg2, "Noisy", (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(label_bg3, "Mean Filter", (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(label_bg4, "Median Filter", (10, 20), font, font_scale, font_color, font_thickness)
    
    # Ensure all images are 3-channel
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(noisy.shape) == 2:
        noisy = cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR)
    if len(mean_filtered.shape) == 2:
        mean_filtered = cv2.cvtColor(mean_filtered, cv2.COLOR_GRAY2BGR)
    if len(median_filtered.shape) == 2:
        median_filtered = cv2.cvtColor(median_filtered, cv2.COLOR_GRAY2BGR)
    
    # Combine labels and images
    top_labels = np.hstack((label_bg1, label_bg2))
    bottom_labels = np.hstack((label_bg3, label_bg4))
    
    comparison = np.vstack((top_labels, top_row, bottom_labels, bottom_row))
    
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_noise_reduction.jpg")
    processor.save_image(comparison, comparison_path)
    
    print(f"Noise reduction test completed.")
    print(f"Results saved to {mean_path}, {gaussian_path}, and {median_path}")
    print(f"Comparison saved to {comparison_path}")
    
    return True

def test_edge_enhancement():
    """Test edge enhancement and sharpening using Sobel operator."""
    print("\n=== Testing Edge Enhancement and Sharpening ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load an image
    image_path = os.path.join(INPUT_DIR, "house.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Convert to grayscale for edge detection
    gray = processor.convert_to_grayscale(original)
    
    # Apply Sobel edge detection
    edges_h = processor.detect_edges_sobel(gray, dx=1, dy=0)
    edges_v = processor.detect_edges_sobel(gray, dx=0, dy=1)
    edges_both = processor.detect_edges_sobel(gray, dx=1, dy=1)
    
    # Save results
    edges_h_path = os.path.join(OUTPUT_DIR, "edges_horizontal_house.jpg")
    edges_v_path = os.path.join(OUTPUT_DIR, "edges_vertical_house.jpg")
    edges_both_path = os.path.join(OUTPUT_DIR, "edges_both_house.jpg")
    
    processor.save_image(edges_h, edges_h_path)
    processor.save_image(edges_v, edges_v_path)
    processor.save_image(edges_both, edges_both_path)
    
    # Apply sharpening
    sharpened = processor.sharpen_image(original, amount=1.5)
    sharpened_path = os.path.join(OUTPUT_DIR, "sharpened_house.jpg")
    processor.save_image(sharpened, sharpened_path)
    
    # Apply Laplacian sharpening
    laplacian_sharpened = processor.laplacian_sharpening(original)
    laplacian_path = os.path.join(OUTPUT_DIR, "laplacian_sharpened_house.jpg")
    processor.save_image(laplacian_sharpened, laplacian_path)
    
    # Create comparison for edge detection
    # Combine all edge images into a grid
    if len(edges_h.shape) == 2:
        edges_h = cv2.cvtColor(edges_h, cv2.COLOR_GRAY2BGR)
    if len(edges_v.shape) == 2:
        edges_v = cv2.cvtColor(edges_v, cv2.COLOR_GRAY2BGR)
    if len(edges_both.shape) == 2:
        edges_both = cv2.cvtColor(edges_both, cv2.COLOR_GRAY2BGR)
    
    edge_comparison = np.hstack((edges_h, edges_v, edges_both))
    edge_comparison_path = os.path.join(OUTPUT_DIR, "comparison_edges.jpg")
    processor.save_image(edge_comparison, edge_comparison_path)
    
    # Create comparison for sharpening
    sharpen_comparison = create_side_by_side_comparison(original, sharpened, ("Original", "Sharpened"))
    sharpen_comparison_path = os.path.join(OUTPUT_DIR, "comparison_sharpening.jpg")
    processor.save_image(sharpen_comparison, sharpen_comparison_path)
    
    print(f"Edge enhancement test completed. Results saved to {edges_h_path}, {edges_v_path}, and {edges_both_path}")
    print(f"Sharpening test completed. Results saved to {sharpened_path} and {laplacian_path}")
    print(f"Comparisons saved to {edge_comparison_path} and {sharpen_comparison_path}")
    
    return True

def test_image_filtering():
    """Test image filtering functionality."""
    print("\n=== Testing Image Filtering ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load an image
    image_path = os.path.join(INPUT_DIR, "mandrill.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Apply bilateral filter
    bilateral = processor.apply_bilateral_filter(original, d=9, sigma_color=75, sigma_space=75)
    bilateral_path = os.path.join(OUTPUT_DIR, "bilateral_mandrill.jpg")
    processor.save_image(bilateral, bilateral_path)
    
    # Apply custom filters
    # Emboss filter
    emboss_kernel = np.array([[-2, -1, 0],
                              [-1, 1, 1],
                              [0, 1, 2]])
    
    emboss = processor.apply_custom_filter(original, emboss_kernel)
    emboss_path = os.path.join(OUTPUT_DIR, "emboss_mandrill.jpg")
    processor.save_image(emboss, emboss_path)
    
    # Edge detection filter
    edge_kernel = np.array([[-1, -1, -1],
                            [-1, 8, -1],
                            [-1, -1, -1]])
    
    edge = processor.apply_custom_filter(original, edge_kernel)
    edge_path = os.path.join(OUTPUT_DIR, "edge_filter_mandrill.jpg")
    processor.save_image(edge, edge_path)
    
    # Create comparison
    # Combine all filtered images into a grid
    comparison = create_side_by_side_comparison(original, bilateral, ("Original", "Bilateral Filter"))
    comparison_path = os.path.join(OUTPUT_DIR, "comparison_filtering.jpg")
    processor.save_image(comparison, comparison_path)
    
    print(f"Image filtering test completed.")
    print(f"Results saved to {bilateral_path}, {emboss_path}, and {edge_path}")
    print(f"Comparison saved to {comparison_path}")
    
    return True

def test_interpolation_magnification():
    """Test interpolation and magnification functionality."""
    print("\n=== Testing Interpolation and Magnification ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load an image
    image_path = os.path.join(INPUT_DIR, "resolution_chart.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Apply different interpolation methods
    nearest = processor.resize_image(original, scale_factor=2.0, interpolation=cv2.INTER_NEAREST)
    linear = processor.resize_image(original, scale_factor=2.0, interpolation=cv2.INTER_LINEAR)
    cubic = processor.resize_image(original, scale_factor=2.0, interpolation=cv2.INTER_CUBIC)
    lanczos = processor.resize_image(original, scale_factor=2.0, interpolation=cv2.INTER_LANCZOS4)
    
    # Save results
    nearest_path = os.path.join(OUTPUT_DIR, "nearest_interpolation.jpg")
    linear_path = os.path.join(OUTPUT_DIR, "linear_interpolation.jpg")
    cubic_path = os.path.join(OUTPUT_DIR, "cubic_interpolation.jpg")
    lanczos_path = os.path.join(OUTPUT_DIR, "lanczos_interpolation.jpg")
    
    processor.save_image(nearest, nearest_path)
    processor.save_image(linear, linear_path)
    processor.save_image(cubic, cubic_path)
    processor.save_image(lanczos, lanczos_path)
    
    # Create comparison
    # Extract a small region for detailed comparison
    h, w = original.shape[:2]
    region = original[h//4:h//2, w//4:w//2]
    
    # Resize the region using different methods
    region_nearest = processor.resize_image(region, scale_factor=4.0, interpolation=cv2.INTER_NEAREST)
    region_linear = processor.resize_image(region, scale_factor=4.0, interpolation=cv2.INTER_LINEAR)
    region_cubic = processor.resize_image(region, scale_factor=4.0, interpolation=cv2.INTER_CUBIC)
    region_lanczos = processor.resize_image(region, scale_factor=4.0, interpolation=cv2.INTER_LANCZOS4)
    
    # Save region results
    region_nearest_path = os.path.join(OUTPUT_DIR, "region_nearest.jpg")
    region_linear_path = os.path.join(OUTPUT_DIR, "region_linear.jpg")
    region_cubic_path = os.path.join(OUTPUT_DIR, "region_cubic.jpg")
    region_lanczos_path = os.path.join(OUTPUT_DIR, "region_lanczos.jpg")
    
    processor.save_image(region_nearest, region_nearest_path)
    processor.save_image(region_linear, region_linear_path)
    processor.save_image(region_cubic, region_cubic_path)
    processor.save_image(region_lanczos, region_lanczos_path)
    
    print(f"Interpolation and magnification test completed.")
    print(f"Results saved to {nearest_path}, {linear_path}, {cubic_path}, and {lanczos_path}")
    print(f"Region results saved to {region_nearest_path}, {region_linear_path}, {region_cubic_path}, and {region_lanczos_path}")
    
    return True

def test_pseudocolor_processing():
    """Test pseudocolor processing functionality."""
    print("\n=== Testing Pseudocolor Processing ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load grayscale images
    image_paths = [
        os.path.join(INPUT_DIR, "moon.tiff"),
        os.path.join(INPUT_DIR, "mri.tiff"),
        os.path.join(INPUT_DIR, "xray.tiff")
    ]
    
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        original = processor.load_image(image_path)
        
        if original is None:
            print(f"Error: Could not load image {image_path}")
            continue
        
        # Ensure image is grayscale
        if len(original.shape) == 3:
            gray = processor.convert_to_grayscale(original)
        else:
            gray = original
        
        # Apply different colormaps
        jet = processor.apply_pseudocolor(gray, colormap=cv2.COLORMAP_JET)
        hot = processor.apply_pseudocolor(gray, colormap=cv2.COLORMAP_HOT)
        rainbow = processor.apply_pseudocolor(gray, colormap=cv2.COLORMAP_RAINBOW)
        viridis = processor.apply_pseudocolor(gray, colormap=cv2.COLORMAP_VIRIDIS)
        
        # Apply custom false color
        false_color = processor.apply_false_color(gray)
        
        # Save results
        jet_path = os.path.join(OUTPUT_DIR, f"{base_name}_jet.jpg")
        hot_path = os.path.join(OUTPUT_DIR, f"{base_name}_hot.jpg")
        rainbow_path = os.path.join(OUTPUT_DIR, f"{base_name}_rainbow.jpg")
        viridis_path = os.path.join(OUTPUT_DIR, f"{base_name}_viridis.jpg")
        false_color_path = os.path.join(OUTPUT_DIR, f"{base_name}_false_color.jpg")
        
        processor.save_image(jet, jet_path)
        processor.save_image(hot, hot_path)
        processor.save_image(rainbow, rainbow_path)
        processor.save_image(viridis, viridis_path)
        processor.save_image(false_color, false_color_path)
        
        # Create comparison
        # Convert grayscale to BGR for comparison
        if len(gray.shape) == 2:
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            gray_bgr = gray
        
        # Create a grid of images
        top_row = np.hstack((gray_bgr, jet, hot))
        bottom_row = np.hstack((rainbow, viridis, false_color))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        font_color = (255, 255, 255)
        
        # Create label backgrounds
        h, w = gray_bgr.shape[:2]
        label_height = 30
        
        label_bg1 = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_bg2 = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_bg3 = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_bg4 = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_bg5 = np.zeros((label_height, w, 3), dtype=np.uint8)
        label_bg6 = np.zeros((label_height, w, 3), dtype=np.uint8)
        
        cv2.putText(label_bg1, "Grayscale", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(label_bg2, "Jet", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(label_bg3, "Hot", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(label_bg4, "Rainbow", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(label_bg5, "Viridis", (10, 20), font, font_scale, font_color, font_thickness)
        cv2.putText(label_bg6, "False Color", (10, 20), font, font_scale, font_color, font_thickness)
        
        # Combine labels and images
        top_labels = np.hstack((label_bg1, label_bg2, label_bg3))
        bottom_labels = np.hstack((label_bg4, label_bg5, label_bg6))
        
        comparison = np.vstack((top_labels, top_row, bottom_labels, bottom_row))
        
        comparison_path = os.path.join(OUTPUT_DIR, f"comparison_pseudocolor_{base_name}.jpg")
        processor.save_image(comparison, comparison_path)
        
        print(f"Pseudocolor processing test completed for {filename}.")
        print(f"Results saved to {jet_path}, {hot_path}, {rainbow_path}, {viridis_path}, and {false_color_path}")
        print(f"Comparison saved to {comparison_path}")
    
    return True

def test_metrics_calculation():
    """Test image metrics calculation."""
    print("\n=== Testing Metrics Calculation ===")
    
    # Initialize processor
    processor = ImageProcessor()
    
    # Load an image
    image_path = os.path.join(INPUT_DIR, "lena.tiff")
    original = processor.load_image(image_path)
    
    if original is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    # Apply different enhancements
    grayscale = processor.convert_to_grayscale(original)
    contrast = processor.adjust_contrast(original, alpha=1.5, beta=10)
    blurred = processor.apply_gaussian_filter(original, kernel_size=5, sigma=1.5)
    sharpened = processor.sharpen_image(original, amount=1.5)
    
    # Calculate metrics for each enhancement
    metrics_grayscale = get_image_metrics(original, grayscale)
    metrics_contrast = get_image_metrics(original, contrast)
    metrics_blurred = get_image_metrics(original, blurred)
    metrics_sharpened = get_image_metrics(original, sharpened)
    
    # Print metrics
    print("\nMetrics for Grayscale Conversion:")
    for name, value in metrics_grayscale.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nMetrics for Contrast Adjustment:")
    for name, value in metrics_contrast.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nMetrics for Gaussian Blur:")
    for name, value in metrics_blurred.items():
        print(f"  {name}: {value:.4f}")
    
    print("\nMetrics for Sharpening:")
    for name, value in metrics_sharpened.items():
        print(f"  {name}: {value:.4f}")
    
    # Create a metrics comparison chart
    plt.figure(figsize=(12, 8))
    
    # PSNR comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Grayscale', 'Contrast', 'Blur', 'Sharpen'], 
            [metrics_grayscale['psnr'], metrics_contrast['psnr'], 
             metrics_blurred['psnr'], metrics_sharpened['psnr']])
    plt.title('PSNR Comparison')
    plt.ylabel('PSNR (dB)')
    plt.grid(alpha=0.3)
    
    # SSIM comparison
    plt.subplot(2, 2, 2)
    plt.bar(['Grayscale', 'Contrast', 'Blur', 'Sharpen'], 
            [metrics_grayscale['ssim'], metrics_contrast['ssim'], 
             metrics_blurred['ssim'], metrics_sharpened['ssim']])
    plt.title('SSIM Comparison')
    plt.ylabel('SSIM')
    plt.grid(alpha=0.3)
    
    # MSE comparison
    plt.subplot(2, 2, 3)
    plt.bar(['Grayscale', 'Contrast', 'Blur', 'Sharpen'], 
            [metrics_grayscale['mse'], metrics_contrast['mse'], 
             metrics_blurred['mse'], metrics_sharpened['mse']])
    plt.title('MSE Comparison')
    plt.ylabel('MSE')
    plt.grid(alpha=0.3)
    
    # Histogram similarity comparison
    plt.subplot(2, 2, 4)
    plt.bar(['Grayscale', 'Contrast', 'Blur', 'Sharpen'], 
            [metrics_grayscale['hist_similarity'], metrics_contrast['hist_similarity'], 
             metrics_blurred['hist_similarity'], metrics_sharpened['hist_similarity']])
    plt.title('Histogram Similarity Comparison')
    plt.ylabel('Similarity')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save the chart
    metrics_chart_path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    plt.savefig(metrics_chart_path)
    plt.close()
    
    print(f"\nMetrics comparison chart saved to {metrics_chart_path}")
    
    return True

def main():
    """Main function to run all tests."""
    print("Starting Image Enhancement System functionality tests...")
    
    # Run tests
    tests = [
        ("Grayscale Conversion", test_grayscale_conversion),
        ("Contrast Adjustment", test_contrast_adjustment),
        ("Noise Reduction", test_noise_reduction),
        ("Edge Enhancement", test_edge_enhancement),
        ("Image Filtering", test_image_filtering),
        ("Interpolation and Magnification", test_interpolation_magnification),
        ("Pseudocolor Processing", test_pseudocolor_processing),
        ("Metrics Calculation", test_metrics_calculation)
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running test: {name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"Error in {name} test: {str(e)}")
            results.append((name, False))
    
    # Print summary
    print("\n\n")
    print("="*50)
    print("Test Results Summary")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "PASSED" if success else "FAILED"
        if not success:
            all_passed = False
        print(f"{name}: {status}")
    
    print("\nAll tests passed!" if all_passed else "\nSome tests failed!")
    print(f"Test results saved to: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
