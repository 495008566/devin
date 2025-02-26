"""
Core image processing module for spatial domain techniques.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, Optional, List, Union

class ImageProcessor:
    """Class for handling image processing operations in the spatial domain."""
    
    def __init__(self):
        """Initialize the image processor."""
        pass
    
    def load_image(self, file_path: str) -> np.ndarray:
        """
        Load an image from file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Loaded image as numpy array
        """
        return cv2.imread(file_path)
    
    def save_image(self, image: np.ndarray, file_path: str) -> bool:
        """
        Save an image to file.
        
        Args:
            image: Image as numpy array
            file_path: Path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        return cv2.imwrite(file_path, image)
    
    def get_image_info(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get basic information about an image.
        
        Args:
            image: Image as numpy array
            
        Returns:
            Dictionary with image information
        """
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        # Calculate basic statistics
        if channels == 1:
            mean = np.mean(image)
            std = np.std(image)
            min_val = np.min(image)
            max_val = np.max(image)
        else:
            mean = [np.mean(image[:,:,i]) for i in range(channels)]
            std = [np.std(image[:,:,i]) for i in range(channels)]
            min_val = [np.min(image[:,:,i]) for i in range(channels)]
            max_val = [np.max(image[:,:,i]) for i in range(channels)]
        
        return {
            'width': width,
            'height': height,
            'channels': channels,
            'size': image.size,
            'dtype': str(image.dtype),
            'mean': mean,
            'std': std,
            'min': min_val,
            'max': max_val
        }
    
    # Grayscale conversion and contrast adjustment
    def convert_to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert an image to grayscale.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 2:
            return image  # Already grayscale
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def adjust_contrast(self, image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
        """
        Adjust image contrast using the formula: g(x,y) = alpha * f(x,y) + beta
        
        Args:
            image: Input image
            alpha: Contrast control (1.0-3.0)
            beta: Brightness control (0-100)
            
        Returns:
            Contrast adjusted image
        """
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using histogram equalization.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Equalized image
        """
        if len(image.shape) == 3:
            # Convert to YUV color space
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            # Apply histogram equalization to the Y channel
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            # Convert back to BGR color space
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        else:
            # Grayscale image
            return cv2.equalizeHist(image)
    
    def adaptive_histogram_equalization(self, image: np.ndarray, clip_limit: float = 2.0, 
                                       tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            CLAHE processed image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            # Convert back to BGR color space
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale image
            return clahe.apply(image)
    
    # Noise reduction using mean filter
    def apply_mean_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply mean filter for noise reduction.
        
        Args:
            image: Input image
            kernel_size: Size of the filter kernel
            
        Returns:
            Filtered image
        """
        return cv2.blur(image, (kernel_size, kernel_size))
    
    def apply_gaussian_filter(self, image: np.ndarray, kernel_size: int = 3, 
                             sigma: float = 0) -> np.ndarray:
        """
        Apply Gaussian filter for noise reduction.
        
        Args:
            image: Input image
            kernel_size: Size of the filter kernel
            sigma: Standard deviation of the Gaussian kernel
            
        Returns:
            Filtered image
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def apply_median_filter(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Apply median filter for noise reduction (good for salt-and-pepper noise).
        
        Args:
            image: Input image
            kernel_size: Size of the filter kernel
            
        Returns:
            Filtered image
        """
        return cv2.medianBlur(image, kernel_size)
    
    # Edge enhancement and sharpening using Sobel operator
    def detect_edges_sobel(self, image: np.ndarray, dx: int = 1, dy: int = 1, 
                          ksize: int = 3) -> np.ndarray:
        """
        Detect edges using Sobel operator.
        
        Args:
            image: Input grayscale image
            dx: Order of derivative in x direction
            dy: Order of derivative in y direction
            ksize: Size of the Sobel kernel
            
        Returns:
            Edge detected image
        """
        gray = self.convert_to_grayscale(image) if len(image.shape) == 3 else image
        sobel = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
        # Convert back to uint8
        abs_sobel = cv2.convertScaleAbs(sobel)
        return abs_sobel
    
    def sharpen_image(self, image: np.ndarray, amount: float = 1.0) -> np.ndarray:
        """
        Sharpen an image using unsharp masking.
        
        Args:
            image: Input image
            amount: Sharpening amount
            
        Returns:
            Sharpened image
        """
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
        return sharpened
    
    def laplacian_sharpening(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """
        Sharpen an image using Laplacian operator.
        
        Args:
            image: Input image
            kernel_size: Size of the Laplacian kernel
            
        Returns:
            Sharpened image
        """
        gray = self.convert_to_grayscale(image) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
        # Convert back to uint8
        abs_laplacian = cv2.convertScaleAbs(laplacian)
        
        if len(image.shape) == 3:
            # For color images, add the laplacian to each channel
            sharpened = image.copy()
            for i in range(3):
                sharpened[:,:,i] = cv2.add(image[:,:,i], abs_laplacian)
            return sharpened
        else:
            # For grayscale images
            return cv2.add(gray, abs_laplacian)
    
    # Image filtering
    def apply_custom_filter(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """
        Apply a custom filter kernel to an image.
        
        Args:
            image: Input image
            kernel: Custom filter kernel
            
        Returns:
            Filtered image
        """
        return cv2.filter2D(image, -1, kernel)
    
    def apply_bilateral_filter(self, image: np.ndarray, d: int = 9, 
                              sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Apply bilateral filter (edge-preserving smoothing).
        
        Args:
            image: Input image
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
            
        Returns:
            Filtered image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    # Interpolation and magnification
    def resize_image(self, image: np.ndarray, scale_factor: float = 2.0, 
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize an image using specified interpolation method.
        
        Args:
            image: Input image
            scale_factor: Scale factor for resizing
            interpolation: Interpolation method
                (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4)
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height), interpolation=interpolation)
    
    # Pseudocolor processing
    def apply_pseudocolor(self, image: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Apply pseudocolor to a grayscale image.
        
        Args:
            image: Input grayscale image
            colormap: OpenCV colormap
                (cv2.COLORMAP_AUTUMN, cv2.COLORMAP_JET, cv2.COLORMAP_HOT, etc.)
            
        Returns:
            Pseudocolored image
        """
        gray = self.convert_to_grayscale(image) if len(image.shape) == 3 else image
        return cv2.applyColorMap(gray, colormap)
    
    def apply_false_color(self, image: np.ndarray) -> np.ndarray:
        """
        Apply false color to a grayscale image by mapping intensity to RGB.
        
        Args:
            image: Input grayscale image
            
        Returns:
            False colored image
        """
        gray = self.convert_to_grayscale(image) if len(image.shape) == 3 else image
        
        # Create an empty color image
        color_img = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        
        # Map intensity ranges to different colors
        # Low intensity (0-85) -> blue
        color_img[:,:,0] = np.where(gray < 85, 255 - gray * 3, 0)
        # Mid intensity (85-170) -> green
        color_img[:,:,1] = np.where((gray >= 85) & (gray < 170), 255 - (gray - 85) * 3, 0)
        # High intensity (170-255) -> red
        color_img[:,:,2] = np.where(gray >= 170, 255 - (gray - 170) * 3, 0)
        
        return color_img
    
    # Image metrics
    def calculate_psnr(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Calculate Peak Signal-to-Noise Ratio between two images.
        
        Args:
            original: Original image
            processed: Processed image
            
        Returns:
            PSNR value in dB
        """
        # Convert to grayscale if needed
        if len(original.shape) == 3 and len(processed.shape) == 3:
            original_gray = self.convert_to_grayscale(original)
            processed_gray = self.convert_to_grayscale(processed)
        else:
            original_gray = original
            processed_gray = processed
            
        return cv2.PSNR(original_gray, processed_gray)
    
    def calculate_histogram(self, image: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Calculate image histogram.
        
        Args:
            image: Input image
            
        Returns:
            Histogram data (single array for grayscale, list of arrays for color)
        """
        if len(image.shape) == 2:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            return hist
        else:
            # Color image
            hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
            return [hist_b, hist_g, hist_r]
    
    def plot_histogram(self, image: np.ndarray, output_path: Optional[str] = None) -> None:
        """
        Plot image histogram.
        
        Args:
            image: Input image
            output_path: Path to save the histogram plot (optional)
        """
        plt.figure(figsize=(10, 6))
        
        if len(image.shape) == 2:
            # Grayscale image
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plt.plot(hist, color='black')
            plt.xlim([0, 256])
            plt.title('Grayscale Histogram')
        else:
            # Color image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
            plt.xlim([0, 256])
            plt.title('Color Histogram')
            
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
