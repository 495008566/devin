"""
Image quality metrics and comparison utilities.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy

def calculate_mse(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error between two images.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        MSE value
    """
    # Ensure images are grayscale
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        
    if len(processed.shape) == 3:
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        processed_gray = processed
    
    # Ensure images are the same size
    if original_gray.shape != processed_gray.shape:
        processed_gray = cv2.resize(processed_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    return np.mean((original_gray.astype(np.float32) - processed_gray.astype(np.float32)) ** 2)

def calculate_psnr(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        PSNR value in dB
    """
    mse = calculate_mse(original, processed)
    if mse == 0:
        return float('inf')
    
    # Assuming 8-bit images (max value 255)
    return 20 * np.log10(255.0) - 10 * np.log10(mse)

def calculate_ssim(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate Structural Similarity Index between two images.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        SSIM value (between -1 and 1, higher is better)
    """
    # Ensure images are grayscale
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        
    if len(processed.shape) == 3:
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        processed_gray = processed
    
    # Ensure images are the same size
    if original_gray.shape != processed_gray.shape:
        processed_gray = cv2.resize(processed_gray, (original_gray.shape[1], original_gray.shape[0]))
    
    return ssim(original_gray, processed_gray)

def calculate_histogram_similarity(original: np.ndarray, processed: np.ndarray) -> float:
    """
    Calculate histogram similarity using correlation method.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        Correlation value (between 0 and 1, higher is better)
    """
    # Ensure images are grayscale
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        
    if len(processed.shape) == 3:
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        processed_gray = processed
    
    # Calculate histograms
    hist1 = cv2.calcHist([original_gray], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([processed_gray], [0], None, [256], [0, 256])
    
    # Normalize histograms
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    
    # Calculate correlation
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate image entropy (measure of randomness).
    
    Args:
        image: Input image
        
    Returns:
        Entropy value
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist / np.sum(hist)  # Normalize
    
    # Calculate entropy
    return entropy(hist[hist > 0])

def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate image contrast using standard deviation.
    
    Args:
        image: Input image
        
    Returns:
        Contrast value
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    return np.std(gray.astype(np.float32))

def get_image_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
    """
    Calculate various image quality metrics.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        Dictionary with metric names and values
    """
    return {
        'mse': calculate_mse(original, processed),
        'psnr': calculate_psnr(original, processed),
        'ssim': calculate_ssim(original, processed),
        'hist_similarity': calculate_histogram_similarity(original, processed),
        'entropy_original': calculate_entropy(original),
        'entropy_processed': calculate_entropy(processed),
        'contrast_original': calculate_contrast(original),
        'contrast_processed': calculate_contrast(processed)
    }
