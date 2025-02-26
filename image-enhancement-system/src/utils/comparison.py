"""
Utility functions for image comparison visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List

def create_side_by_side_comparison(original: np.ndarray, processed: np.ndarray, 
                                  labels: Tuple[str, str] = ('Original', 'Processed')) -> np.ndarray:
    """
    Create a side-by-side comparison of two images.
    
    Args:
        original: Original image
        processed: Processed image
        labels: Labels for the images
        
    Returns:
        Combined image with both original and processed
    """
    # Ensure images are the same size
    if original.shape[:2] != processed.shape[:2]:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Ensure both images have the same number of channels
    if len(original.shape) != len(processed.shape):
        if len(original.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        else:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    
    # Create a black separator
    separator_width = 5
    separator = np.zeros((original.shape[0], separator_width, 3), dtype=np.uint8)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    font_color = (255, 255, 255)
    
    # Create label backgrounds
    label_height = 30
    label_bg1 = np.zeros((label_height, original.shape[1], 3), dtype=np.uint8)
    label_bg2 = np.zeros((label_height, processed.shape[1], 3), dtype=np.uint8)
    
    # Add text to label backgrounds
    cv2.putText(label_bg1, labels[0], (10, 20), font, font_scale, font_color, font_thickness)
    cv2.putText(label_bg2, labels[1], (10, 20), font, font_scale, font_color, font_thickness)
    
    # Ensure original and processed are 3-channel
    if len(original.shape) == 2:
        original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    
    # Combine images horizontally with labels
    top_row = np.hstack((label_bg1, np.zeros((label_height, separator_width, 3), dtype=np.uint8), label_bg2))
    bottom_row = np.hstack((original, separator, processed))
    
    return np.vstack((top_row, bottom_row))

def create_before_after_slider(original: np.ndarray, processed: np.ndarray, 
                              output_path: str) -> None:
    """
    Create an interactive before/after comparison and save as HTML.
    
    Args:
        original: Original image
        processed: Processed image
        output_path: Path to save the HTML file
    """
    # Ensure images are the same size
    if original.shape[:2] != processed.shape[:2]:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert BGR to RGB for matplotlib
    if len(original.shape) == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
        
    if len(processed.shape) == 3:
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    else:
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    
    # Create HTML with JavaScript for slider
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Before/After Comparison</title>
        <style>
            .comparison-slider {{
                position: relative;
                width: 100%;
                max-width: 800px;
                overflow: hidden;
                margin: 0 auto;
            }}
            .comparison-slider img {{
                width: 100%;
                display: block;
            }}
            .comparison-slider .img-overlay {{
                position: absolute;
                top: 0;
                left: 0;
                height: 100%;
                width: 50%;
                overflow: hidden;
            }}
            .comparison-slider .slider-handle {{
                position: absolute;
                top: 0;
                bottom: 0;
                left: 50%;
                width: 4px;
                background: white;
                cursor: ew-resize;
            }}
            .comparison-slider .slider-handle::after {{
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                width: 30px;
                height: 30px;
                border-radius: 50%;
                background: white;
                border: 4px solid #333;
            }}
            .comparison-slider .label {{
                position: absolute;
                background: rgba(0, 0, 0, 0.5);
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-family: Arial, sans-serif;
            }}
            .comparison-slider .label.before {{
                top: 10px;
                left: 10px;
            }}
            .comparison-slider .label.after {{
                top: 10px;
                right: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="comparison-slider">
            <img src="data:image/png;base64,PROCESSED_IMAGE" alt="After">
            <div class="img-overlay">
                <img src="data:image/png;base64,ORIGINAL_IMAGE" alt="Before">
            </div>
            <div class="slider-handle"></div>
            <div class="label before">Before</div>
            <div class="label after">After</div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                const slider = document.querySelector('.comparison-slider');
                const handle = slider.querySelector('.slider-handle');
                const overlay = slider.querySelector('.img-overlay');
                let isDragging = false;
                
                const move = (e) => {{
                    if (!isDragging) return;
                    
                    const sliderRect = slider.getBoundingClientRect();
                    const x = e.clientX - sliderRect.left;
                    const position = Math.max(0, Math.min(x / sliderRect.width, 1));
                    
                    overlay.style.width = position * 100 + '%';
                    handle.style.left = position * 100 + '%';
                }};
                
                handle.addEventListener('mousedown', () => {{
                    isDragging = true;
                }});
                
                window.addEventListener('mouseup', () => {{
                    isDragging = false;
                }});
                
                window.addEventListener('mousemove', move);
                
                // Touch support
                handle.addEventListener('touchstart', (e) => {{
                    isDragging = true;
                    e.preventDefault();
                }});
                
                window.addEventListener('touchend', () => {{
                    isDragging = false;
                }});
                
                window.addEventListener('touchmove', (e) => {{
                    if (!isDragging) return;
                    
                    const touch = e.touches[0];
                    const sliderRect = slider.getBoundingClientRect();
                    const x = touch.clientX - sliderRect.left;
                    const position = Math.max(0, Math.min(x / sliderRect.width, 1));
                    
                    overlay.style.width = position * 100 + '%';
                    handle.style.left = position * 100 + '%';
                    
                    e.preventDefault();
                }});
            }});
        </script>
    </body>
    </html>
    """
    
    # Convert images to base64
    _, original_buffer = cv2.imencode('.png', original_rgb)
    _, processed_buffer = cv2.imencode('.png', processed_rgb)
    
    import base64
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')
    processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
    
    # Replace placeholders with actual base64 images
    html_content = html_content.replace('ORIGINAL_IMAGE', original_base64)
    html_content = html_content.replace('PROCESSED_IMAGE', processed_base64)
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(html_content)

def plot_histograms(original: np.ndarray, processed: np.ndarray, 
                   output_path: Optional[str] = None) -> None:
    """
    Plot histograms of original and processed images for comparison.
    
    Args:
        original: Original image
        processed: Processed image
        output_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Original image histogram
    plt.subplot(2, 2, 1)
    plt.title('Original Image')
    if len(original.shape) == 3:
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title('Original Histogram')
    if len(original.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([original], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
    else:
        hist = cv2.calcHist([original], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.grid(alpha=0.3)
    
    # Processed image histogram
    plt.subplot(2, 2, 3)
    plt.title('Processed Image')
    if len(processed.shape) == 3:
        plt.imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(processed, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title('Processed Histogram')
    if len(processed.shape) == 3:
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([processed], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
    else:
        hist = cv2.calcHist([processed], [0], None, [256], [0, 256])
        plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
