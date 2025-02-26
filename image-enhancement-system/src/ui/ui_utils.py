"""
UI utility functions for the Image Enhancement System.
"""

import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Tuple

def convert_cv_to_tk(image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> ImageTk.PhotoImage:
    """
    Convert an OpenCV image to a Tkinter PhotoImage.
    
    Args:
        image: OpenCV image (numpy array)
        size: Optional size to resize to (width, height)
        
    Returns:
        Tkinter PhotoImage
    """
    # Convert color space if needed
    if len(image.shape) == 3:
        # BGR to RGB
        display_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Grayscale to RGB
        display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Resize if needed
    if size is not None:
        display_img = cv2.resize(display_img, size)
    
    # Convert to PIL Image and then to PhotoImage
    pil_img = Image.fromarray(display_img)
    return ImageTk.PhotoImage(pil_img)

def calculate_display_size(image_shape: Tuple[int, int], canvas_size: Tuple[int, int]) -> Tuple[int, int]:
    """
    Calculate the display size to fit an image in a canvas while maintaining aspect ratio.
    
    Args:
        image_shape: Image shape (height, width)
        canvas_size: Canvas size (width, height)
        
    Returns:
        Display size (width, height)
    """
    img_height, img_width = image_shape[:2]
    canvas_width, canvas_height = canvas_size
    
    # Calculate scaling factor
    scale_width = canvas_width / img_width
    scale_height = canvas_height / img_height
    scale = min(scale_width, scale_height)
    
    # Calculate new dimensions
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    
    return (new_width, new_height)
