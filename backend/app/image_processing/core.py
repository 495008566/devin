import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
import time
from skimage import color

class ImageMetrics:
    """Class to calculate and store image quality metrics."""
    
    @staticmethod
    def calculate_metrics(original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """
        Calculate various image quality metrics between original and processed images.
        
        Args:
            original: Original image as numpy array
            processed: Processed image as numpy array
            
        Returns:
            Dictionary containing metrics
        """
        # Ensure images are in the same format for comparison
        if len(original.shape) != len(processed.shape):
            if len(original.shape) == 3 and len(processed.shape) == 2:
                # Convert grayscale processed image to 3-channel for comparison
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            elif len(original.shape) == 2 and len(processed.shape) == 3:
                # Convert original to grayscale for comparison
                original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        metrics = {}
        
        # Mean Squared Error (MSE)
        if original.shape == processed.shape:
            err = np.sum((original.astype("float") - processed.astype("float")) ** 2)
            err /= float(original.shape[0] * original.shape[1] * (3 if len(original.shape) > 2 else 1))
            metrics['mse'] = err
            
            # Peak Signal-to-Noise Ratio (PSNR)
            if err > 0:
                metrics['psnr'] = 10 * np.log10(255**2 / err)
            else:
                metrics['psnr'] = float('inf')
        
        # Calculate histogram-based metrics
        if len(processed.shape) == 3:
            processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            processed_gray = processed
            
        # Entropy (measure of information content)
        hist = cv2.calcHist([processed_gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        non_zero_hist = hist[hist > 0]
        metrics['entropy'] = -np.sum(non_zero_hist * np.log2(non_zero_hist))
        
        # Contrast (standard deviation of pixel values)
        metrics['contrast'] = np.std(processed_gray)
        
        # Processing time is added by each processing function
        
        return metrics


class ImageProcessor:
    """Base class for image processing operations."""
    
    @staticmethod
    def load_image(image_data: bytes) -> np.ndarray:
        """
        Load image from bytes data.
        
        Args:
            image_data: Image data as bytes
            
        Returns:
            Image as numpy array
        """
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image data")
        return img
    
    @staticmethod
    def encode_image(image: np.ndarray, format: str = '.jpg') -> bytes:
        """
        Encode image to bytes.
        
        Args:
            image: Image as numpy array
            format: Image format extension (e.g., '.jpg', '.png')
            
        Returns:
            Image data as bytes
        """
        success, encoded_img = cv2.imencode(format, image)
        if not success:
            raise ValueError("Failed to encode image")
        return encoded_img.tobytes()
    
    @staticmethod
    def process_image(image: np.ndarray, operation: str, params: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process image with the specified operation.
        
        Args:
            image: Input image as numpy array
            operation: Name of the operation to perform
            params: Parameters for the operation
            
        Returns:
            Tuple of (processed image, metrics dictionary)
        """
        if params is None:
            params = {}
            
        # Create a copy of the original image to avoid modifying it
        original = image.copy()
        
        start_time = time.time()
        
        # Select the appropriate processing function based on the operation
        if operation == "grayscale":
            processed = GrayscaleConverter.convert(image, **params)
        elif operation == "contrast":
            processed = ContrastAdjuster.adjust(image, **params)
        elif operation == "noise_reduction":
            processed = NoiseReducer.reduce(image, **params)
        elif operation == "edge_enhancement":
            processed = EdgeEnhancer.enhance(image, **params)
        elif operation == "sharpening":
            processed = Sharpener.sharpen(image, **params)
        elif operation == "interpolation":
            processed = Interpolator.interpolate(image, **params)
        elif operation == "false_color":
            processed = FalseColorEnhancer.enhance(image, **params)
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Calculate metrics
        metrics = ImageMetrics.calculate_metrics(original, processed)
        metrics['processing_time'] = processing_time
        
        return processed, metrics


class GrayscaleConverter:
    """Class for grayscale conversion operations."""
    
    @staticmethod
    def convert(image: np.ndarray, method: str = 'weighted') -> np.ndarray:
        """
        Convert image to grayscale.
        
        Args:
            image: Input image as numpy array
            method: Conversion method ('weighted', 'average', 'luminosity')
            
        Returns:
            Grayscale image
        """
        if len(image.shape) < 3 or image.shape[2] == 1:
            return image  # Already grayscale
            
        if method == 'weighted':
            # OpenCV default (weighted method: 0.299*R + 0.587*G + 0.114*B)
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif method == 'average':
            # Simple average of channels
            return np.mean(image, axis=2).astype(np.uint8)
        elif method == 'luminosity':
            # Luminosity method (0.21*R + 0.72*G + 0.07*B)
            return np.dot(image[..., :3], [0.07, 0.72, 0.21]).astype(np.uint8)
        else:
            raise ValueError(f"Unknown grayscale method: {method}")


class ContrastAdjuster:
    """Class for contrast adjustment operations."""
    
    @staticmethod
    def adjust(image: np.ndarray, method: str = 'histogram_equalization', 
               alpha: float = 1.5, beta: float = 0, 
               clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image as numpy array
            method: Adjustment method ('histogram_equalization', 'clahe', 'linear')
            alpha: Contrast control (gain) for linear adjustment
            beta: Brightness control (bias) for linear adjustment
            clip_limit: Clip limit for CLAHE
            tile_grid_size: Tile grid size for CLAHE
            
        Returns:
            Contrast-adjusted image
        """
        # Make a copy to avoid modifying the original
        result = image.copy()
        
        if method == 'histogram_equalization':
            if len(image.shape) == 3:
                # Convert to YCrCb color space
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                # Apply histogram equalization to the Y channel
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                # Convert back to BGR
                result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                # For grayscale images
                result = cv2.equalizeHist(image)
                
        elif method == 'clahe':
            # Create CLAHE object
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                # Apply CLAHE to L channel
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                # Convert back to BGR
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                # For grayscale images
                result = clahe.apply(image)
                
        elif method == 'linear':
            # Linear contrast adjustment using alpha (gain) and beta (bias)
            result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        else:
            raise ValueError(f"Unknown contrast adjustment method: {method}")
            
        return result


class NoiseReducer:
    """Class for noise reduction operations."""
    
    @staticmethod
    def reduce(image: np.ndarray, method: str = 'mean', kernel_size: int = 3, 
               strength: int = 7, color_strength: int = 7, template_size: int = 7, 
               search_size: int = 21) -> np.ndarray:
        """
        Reduce noise in image.
        
        Args:
            image: Input image as numpy array
            method: Noise reduction method ('mean', 'gaussian', 'median', 'bilateral', 'nlm')
            kernel_size: Kernel size for filtering
            strength: Filter strength parameter
            color_strength: Color filter strength for bilateral filter
            template_size: Template window size for non-local means
            search_size: Search window size for non-local means
            
        Returns:
            Noise-reduced image
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        if method == 'mean':
            # Mean filter (box filter)
            return cv2.blur(image, (kernel_size, kernel_size))
            
        elif method == 'gaussian':
            # Gaussian filter
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif method == 'median':
            # Median filter
            return cv2.medianBlur(image, kernel_size)
            
        elif method == 'bilateral':
            # Bilateral filter (edge-preserving)
            return cv2.bilateralFilter(image, kernel_size, color_strength, strength)
            
        elif method == 'nlm':
            # Non-local means filter (good for preserving details)
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(
                    image, None, strength, color_strength, template_size, search_size
                )
            else:
                return cv2.fastNlMeansDenoising(
                    image, None, strength, template_size, search_size
                )
                
        else:
            raise ValueError(f"Unknown noise reduction method: {method}")


class EdgeEnhancer:
    """Class for edge enhancement operations."""
    
    @staticmethod
    def enhance(image: np.ndarray, method: str = 'sobel', kernel_size: int = 3, 
                scale: float = 1.0, delta: float = 0, 
                low_threshold: int = 50, high_threshold: int = 150) -> np.ndarray:
        """
        Enhance edges in image.
        
        Args:
            image: Input image as numpy array
            method: Edge detection method ('sobel', 'laplacian', 'canny')
            kernel_size: Kernel size for Sobel operator
            scale: Scale factor for edge detection
            delta: Delta value added to results
            low_threshold: Low threshold for Canny edge detector
            high_threshold: High threshold for Canny edge detector
            
        Returns:
            Edge-enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        if method == 'sobel':
            # Sobel operator
            grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
            grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
            
            # Convert to absolute values
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            
            # Combine gradients
            result = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
            
        elif method == 'laplacian':
            # Laplacian operator
            laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=kernel_size, scale=scale, delta=delta)
            result = cv2.convertScaleAbs(laplacian)
            
        elif method == 'canny':
            # Canny edge detector
            result = cv2.Canny(gray, low_threshold, high_threshold)
            
        else:
            raise ValueError(f"Unknown edge enhancement method: {method}")
            
        # Always convert result back to color if it's grayscale
        if len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            
        return result


class Sharpener:
    """Class for image sharpening operations."""
    
    @staticmethod
    def sharpen(image: np.ndarray, method: str = 'unsharp_mask', 
                kernel_size: int = 3, strength: float = 1.5, 
                laplacian_kernel_size: int = 3) -> np.ndarray:
        """
        Sharpen image.
        
        Args:
            image: Input image as numpy array
            method: Sharpening method ('unsharp_mask', 'laplacian')
            kernel_size: Kernel size for Gaussian blur in unsharp mask
            strength: Sharpening strength
            laplacian_kernel_size: Kernel size for Laplacian operator
            
        Returns:
            Sharpened image
        """
        if method == 'unsharp_mask':
            # Unsharp mask: subtract blurred image from original
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
            return sharpened
            
        elif method == 'laplacian':
            # Laplacian sharpening
            if len(image.shape) == 3:
                # Process each channel separately for color images
                result = np.zeros_like(image)
                for i in range(3):
                    laplacian = cv2.Laplacian(image[:, :, i], cv2.CV_16S, 
                                             ksize=laplacian_kernel_size)
                    result[:, :, i] = cv2.convertScaleAbs(image[:, :, i] + strength * laplacian)
                return result
            else:
                # For grayscale images
                laplacian = cv2.Laplacian(image, cv2.CV_16S, ksize=laplacian_kernel_size)
                return cv2.convertScaleAbs(image + strength * laplacian)
                
        else:
            raise ValueError(f"Unknown sharpening method: {method}")


class Interpolator:
    """Class for image interpolation and magnification operations."""
    
    @staticmethod
    def interpolate(image: np.ndarray, scale_factor: float = 2.0, 
                   method: str = 'bicubic') -> np.ndarray:
        """
        Interpolate (resize) image.
        
        Args:
            image: Input image as numpy array
            scale_factor: Scale factor for resizing
            method: Interpolation method ('nearest', 'bilinear', 'bicubic', 'lanczos')
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Map method names to OpenCV interpolation constants
        interpolation_methods = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4
        }
        
        if method not in interpolation_methods:
            raise ValueError(f"Unknown interpolation method: {method}")
            
        return cv2.resize(image, (new_width, new_height), 
                         interpolation=interpolation_methods[method])


class FalseColorEnhancer:
    """Class for false color enhancement operations."""
    
    @staticmethod
    def enhance(image: np.ndarray, method: str = 'jet', 
               min_value: Optional[float] = None, 
               max_value: Optional[float] = None) -> np.ndarray:
        """
        Apply false color enhancement to image.
        
        Args:
            image: Input image as numpy array
            method: Colormap method ('jet', 'hot', 'cool', 'rainbow', 'viridis', 'plasma')
            min_value: Minimum value for normalization (default: image min)
            max_value: Maximum value for normalization (default: image max)
            
        Returns:
            False color enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Normalize image to 0-255 range if min/max provided
        if min_value is not None and max_value is not None:
            gray_norm = np.clip((gray - min_value) / (max_value - min_value) * 255, 0, 255).astype(np.uint8)
        else:
            gray_norm = gray
            
        # Map colormap names to OpenCV colormap constants
        colormap_methods = {
            'jet': cv2.COLORMAP_JET,
            'hot': cv2.COLORMAP_HOT,
            'cool': cv2.COLORMAP_COOL,
            'rainbow': cv2.COLORMAP_RAINBOW,
            'viridis': cv2.COLORMAP_VIRIDIS,
            'plasma': cv2.COLORMAP_PLASMA
        }
        
        if method not in colormap_methods:
            raise ValueError(f"Unknown colormap method: {method}")
            
        return cv2.applyColorMap(gray_norm, colormap_methods[method])
