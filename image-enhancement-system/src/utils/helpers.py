"""
Utility functions for the Image Enhancement System.
"""

import os
import cv2
import numpy as np
from typing import Dict, Any, Tuple, List

def create_directory_if_not_exists(directory: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File extension
    """
    return os.path.splitext(file_path)[1].lower()

def is_valid_image_file(file_path: str) -> bool:
    """
    Check if a file is a valid image file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if valid image file, False otherwise
    """
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    return get_file_extension(file_path) in valid_extensions
