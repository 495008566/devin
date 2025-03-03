import cv2
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import base64
import io
from PIL import Image

def encode_image_to_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """
    Encode OpenCV image to base64 string.
    
    Args:
        image: OpenCV image (numpy array)
        format: Image format (jpeg, png)
        
    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image)
    
    # Save image to bytes buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    
    # Encode bytes to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/{format};base64,{img_str}"

def decode_base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image.
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        OpenCV image (numpy array)
    """
    # Remove data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',', 1)[1]
    
    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode base64 image")
    
    return img

def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metric values
        
    Returns:
        Dictionary of formatted metric strings
    """
    formatted = {}
    
    for key, value in metrics.items():
        if key == 'processing_time':
            formatted[key] = f"{value:.3f} seconds"
        elif key == 'mse':
            formatted[key] = f"{value:.2f}"
        elif key == 'psnr':
            if value == float('inf'):
                formatted[key] = "∞ dB"
            else:
                formatted[key] = f"{value:.2f} dB"
        elif key == 'entropy':
            formatted[key] = f"{value:.2f} bits"
        elif key == 'contrast':
            formatted[key] = f"{value:.2f}"
        else:
            formatted[key] = f"{value}"
    
    return formatted

def get_available_operations() -> List[Dict[str, Any]]:
    """
    Get list of available image processing operations.
    
    Returns:
        List of operation dictionaries with name, description, and parameters
    """
    return [
        {
            "id": "grayscale",
            "name": "Grayscale Conversion",
            "description": "Convert image to grayscale using different methods",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["weighted", "average", "luminosity"],
                    "default": "weighted",
                    "description": "Grayscale conversion method"
                }
            ]
        },
        {
            "id": "contrast",
            "name": "Contrast Adjustment",
            "description": "Adjust image contrast using different methods",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["histogram_equalization", "clahe", "linear"],
                    "default": "histogram_equalization",
                    "description": "Contrast adjustment method"
                },
                {
                    "name": "alpha",
                    "type": "range",
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "default": 1.5,
                    "description": "Contrast control (gain) for linear adjustment"
                },
                {
                    "name": "beta",
                    "type": "range",
                    "min": -50,
                    "max": 50,
                    "step": 1,
                    "default": 0,
                    "description": "Brightness control (bias) for linear adjustment"
                },
                {
                    "name": "clip_limit",
                    "type": "range",
                    "min": 1.0,
                    "max": 5.0,
                    "step": 0.1,
                    "default": 2.0,
                    "description": "Clip limit for CLAHE"
                }
            ]
        },
        {
            "id": "noise_reduction",
            "name": "Noise Reduction",
            "description": "Reduce noise in image using different filters",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["mean", "gaussian", "median", "bilateral", "nlm"],
                    "default": "mean",
                    "description": "Noise reduction method"
                },
                {
                    "name": "kernel_size",
                    "type": "select",
                    "options": [3, 5, 7, 9],
                    "default": 3,
                    "description": "Kernel size for filtering"
                },
                {
                    "name": "strength",
                    "type": "range",
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "default": 7,
                    "description": "Filter strength parameter"
                }
            ]
        },
        {
            "id": "edge_enhancement",
            "name": "Edge Enhancement",
            "description": "Enhance edges in image using different methods",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["sobel", "laplacian", "canny"],
                    "default": "sobel",
                    "description": "Edge detection method"
                },
                {
                    "name": "kernel_size",
                    "type": "select",
                    "options": [3, 5, 7],
                    "default": 3,
                    "description": "Kernel size for operators"
                },
                {
                    "name": "scale",
                    "type": "range",
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1,
                    "default": 1.0,
                    "description": "Scale factor for edge detection"
                }
            ]
        },
        {
            "id": "sharpening",
            "name": "Image Sharpening",
            "description": "Sharpen image using different methods",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["unsharp_mask", "laplacian"],
                    "default": "unsharp_mask",
                    "description": "Sharpening method"
                },
                {
                    "name": "strength",
                    "type": "range",
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1,
                    "default": 1.5,
                    "description": "Sharpening strength"
                },
                {
                    "name": "kernel_size",
                    "type": "select",
                    "options": [3, 5, 7],
                    "default": 3,
                    "description": "Kernel size for blur in unsharp mask"
                }
            ]
        },
        {
            "id": "interpolation",
            "name": "Interpolation & Magnification",
            "description": "Resize image using different interpolation methods",
            "parameters": [
                {
                    "name": "scale_factor",
                    "type": "range",
                    "min": 0.5,
                    "max": 4.0,
                    "step": 0.1,
                    "default": 2.0,
                    "description": "Scale factor for resizing"
                },
                {
                    "name": "method",
                    "type": "select",
                    "options": ["nearest", "bilinear", "bicubic", "lanczos"],
                    "default": "bicubic",
                    "description": "Interpolation method"
                }
            ]
        },
        {
            "id": "false_color",
            "name": "False Color Enhancement",
            "description": "Apply false color mapping to image",
            "parameters": [
                {
                    "name": "method",
                    "type": "select",
                    "options": ["jet", "hot", "cool", "rainbow", "viridis", "plasma"],
                    "default": "jet",
                    "description": "Colormap method"
                }
            ]
        }
    ]

def optimize_for_large_images(image: np.ndarray, max_size: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Optimize image for processing by resizing if too large.
    
    Args:
        image: Input image
        max_size: Maximum dimension size
        
    Returns:
        Tuple of (resized image, scale factor)
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)
    
    # If image is already small enough, return original
    if max_dim <= max_size:
        return image, 1.0
    
    # Calculate scale factor
    scale = max_size / max_dim
    
    # Resize image
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized, scale

def restore_original_size(image: np.ndarray, original_shape: Tuple[int, int], 
                         interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """
    Restore image to original size after processing.
    
    Args:
        image: Processed image
        original_shape: Original image shape (height, width)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    return cv2.resize(image, (original_shape[1], original_shape[0]), 
                     interpolation=interpolation)
