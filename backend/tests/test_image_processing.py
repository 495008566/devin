import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.image_processing.core import (
    ImageProcessor, GrayscaleConverter, ContrastAdjuster,
    NoiseReducer, EdgeEnhancer, Sharpener, Interpolator,
    FalseColorEnhancer, ImageMetrics
)

class TestImageProcessing(unittest.TestCase):
    """Test cases for image processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image (100x100 with a white square in the middle)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = 255  # White square
        
        # Create a grayscale test image
        self.gray_image = np.zeros((100, 100), dtype=np.uint8)
        self.gray_image[25:75, 25:75] = 255  # White square
        
        # Create a noisy image
        self.noisy_image = self.test_image.copy()
        noise = np.random.normal(0, 25, self.test_image.shape).astype(np.uint8)
        self.noisy_image = cv2.add(self.noisy_image, noise)
        
        # Create a blurry image
        self.blurry_image = cv2.GaussianBlur(self.test_image, (15, 15), 0)

    def test_image_processor_load_encode(self):
        """Test image loading and encoding."""
        # Encode the test image
        encoded = ImageProcessor.encode_image(self.test_image)
        self.assertIsInstance(encoded, bytes)
        
        # Load the encoded image
        decoded = ImageProcessor.load_image(encoded)
        self.assertIsInstance(decoded, np.ndarray)
        
        # Check dimensions
        self.assertEqual(decoded.shape, self.test_image.shape)
        
        # Check content (should be approximately the same)
        diff = np.sum(np.abs(decoded.astype(int) - self.test_image.astype(int)))
        self.assertLess(diff / (decoded.shape[0] * decoded.shape[1] * decoded.shape[2]), 5)

    def test_grayscale_conversion(self):
        """Test grayscale conversion methods."""
        # Test weighted method
        weighted = GrayscaleConverter.convert(self.test_image, method='weighted')
        self.assertEqual(weighted.shape, (100, 100))
        self.assertEqual(weighted.dtype, np.uint8)
        
        # Test average method
        average = GrayscaleConverter.convert(self.test_image, method='average')
        self.assertEqual(average.shape, (100, 100))
        self.assertEqual(average.dtype, np.uint8)
        
        # Test luminosity method
        luminosity = GrayscaleConverter.convert(self.test_image, method='luminosity')
        self.assertEqual(luminosity.shape, (100, 100))
        self.assertEqual(luminosity.dtype, np.uint8)
        
        # Test with already grayscale image
        gray_result = GrayscaleConverter.convert(self.gray_image)
        self.assertEqual(gray_result.shape, self.gray_image.shape)
        np.testing.assert_array_equal(gray_result, self.gray_image)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            GrayscaleConverter.convert(self.test_image, method='invalid')

    def test_contrast_adjustment(self):
        """Test contrast adjustment methods."""
        # Test histogram equalization
        hist_eq = ContrastAdjuster.adjust(self.test_image, method='histogram_equalization')
        self.assertEqual(hist_eq.shape, self.test_image.shape)
        self.assertEqual(hist_eq.dtype, np.uint8)
        
        # Test CLAHE
        clahe = ContrastAdjuster.adjust(self.test_image, method='clahe')
        self.assertEqual(clahe.shape, self.test_image.shape)
        self.assertEqual(clahe.dtype, np.uint8)
        
        # Test linear adjustment
        linear = ContrastAdjuster.adjust(self.test_image, method='linear', alpha=1.5, beta=10)
        self.assertEqual(linear.shape, self.test_image.shape)
        self.assertEqual(linear.dtype, np.uint8)
        
        # Test with grayscale image
        gray_hist_eq = ContrastAdjuster.adjust(self.gray_image, method='histogram_equalization')
        self.assertEqual(gray_hist_eq.shape, self.gray_image.shape)
        self.assertEqual(gray_hist_eq.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            ContrastAdjuster.adjust(self.test_image, method='invalid')

    def test_noise_reduction(self):
        """Test noise reduction methods."""
        # Test mean filter
        mean_filtered = NoiseReducer.reduce(self.noisy_image, method='mean', kernel_size=3)
        self.assertEqual(mean_filtered.shape, self.noisy_image.shape)
        self.assertEqual(mean_filtered.dtype, np.uint8)
        
        # Verify mean filter implementation
        # Create a simple 3x3 kernel and apply it manually to a small region
        kernel_size = 3
        test_region = self.noisy_image[40:43, 40:43].copy()
        manual_mean = np.zeros_like(test_region)
        
        for c in range(test_region.shape[2]):
            channel_sum = np.sum(test_region[:, :, c])
            manual_mean[:, :, c] = channel_sum // 9  # Integer division for uint8
        
        # Apply the mean filter to the same region
        opencv_mean = cv2.blur(test_region, (kernel_size, kernel_size))
        
        # Compare results (allowing for small differences due to border handling)
        diff = np.abs(manual_mean.astype(int) - opencv_mean.astype(int))
        self.assertLess(np.mean(diff), 10)
        
        # Test gaussian filter
        gaussian = NoiseReducer.reduce(self.noisy_image, method='gaussian', kernel_size=3)
        self.assertEqual(gaussian.shape, self.noisy_image.shape)
        self.assertEqual(gaussian.dtype, np.uint8)
        
        # Test median filter
        median = NoiseReducer.reduce(self.noisy_image, method='median', kernel_size=3)
        self.assertEqual(median.shape, self.noisy_image.shape)
        self.assertEqual(median.dtype, np.uint8)
        
        # Test bilateral filter
        bilateral = NoiseReducer.reduce(self.noisy_image, method='bilateral', kernel_size=3)
        self.assertEqual(bilateral.shape, self.noisy_image.shape)
        self.assertEqual(bilateral.dtype, np.uint8)
        
        # Test non-local means filter
        nlm = NoiseReducer.reduce(self.noisy_image, method='nlm')
        self.assertEqual(nlm.shape, self.noisy_image.shape)
        self.assertEqual(nlm.dtype, np.uint8)
        
        # Test with grayscale image
        gray_mean = NoiseReducer.reduce(cv2.cvtColor(self.noisy_image, cv2.COLOR_BGR2GRAY), method='mean')
        self.assertEqual(len(gray_mean.shape), 2)
        self.assertEqual(gray_mean.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            NoiseReducer.reduce(self.noisy_image, method='invalid')

    def test_edge_enhancement(self):
        """Test edge enhancement methods."""
        # Test Sobel operator
        sobel = EdgeEnhancer.enhance(self.test_image, method='sobel')
        self.assertEqual(sobel.shape, self.test_image.shape)
        self.assertEqual(sobel.dtype, np.uint8)
        
        # Verify Sobel operator implementation
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        manual_sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Convert manual_sobel to 3-channel for comparison
        manual_sobel_bgr = cv2.cvtColor(manual_sobel, cv2.COLOR_GRAY2BGR)
        
        # Compare results (allowing for small differences due to implementation details)
        diff = np.abs(manual_sobel_bgr.astype(int) - sobel.astype(int))
        self.assertLess(np.mean(diff), 5)
        
        # Test Laplacian operator
        laplacian = EdgeEnhancer.enhance(self.test_image, method='laplacian')
        self.assertEqual(laplacian.shape, self.test_image.shape)
        self.assertEqual(laplacian.dtype, np.uint8)
        
        # Test Canny edge detector
        canny = EdgeEnhancer.enhance(self.test_image, method='canny')
        self.assertEqual(canny.shape, self.test_image.shape)
        self.assertEqual(canny.dtype, np.uint8)
        
        # Test with grayscale image
        gray_sobel = EdgeEnhancer.enhance(self.gray_image, method='sobel')
        self.assertEqual(len(gray_sobel.shape), 3)  # Should be converted to BGR
        self.assertEqual(gray_sobel.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            EdgeEnhancer.enhance(self.test_image, method='invalid')

    def test_sharpening(self):
        """Test image sharpening methods."""
        # Test unsharp mask
        unsharp = Sharpener.sharpen(self.blurry_image, method='unsharp_mask')
        self.assertEqual(unsharp.shape, self.blurry_image.shape)
        self.assertEqual(unsharp.dtype, np.uint8)
        
        # Test Laplacian sharpening
        laplacian = Sharpener.sharpen(self.blurry_image, method='laplacian')
        self.assertEqual(laplacian.shape, self.blurry_image.shape)
        self.assertEqual(laplacian.dtype, np.uint8)
        
        # Test with grayscale image
        gray_unsharp = Sharpener.sharpen(cv2.cvtColor(self.blurry_image, cv2.COLOR_BGR2GRAY), method='unsharp_mask')
        self.assertEqual(len(gray_unsharp.shape), 2)
        self.assertEqual(gray_unsharp.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            Sharpener.sharpen(self.blurry_image, method='invalid')

    def test_interpolation(self):
        """Test image interpolation methods."""
        # Test nearest neighbor
        nearest = Interpolator.interpolate(self.test_image, scale_factor=2.0, method='nearest')
        self.assertEqual(nearest.shape, (200, 200, 3))
        self.assertEqual(nearest.dtype, np.uint8)
        
        # Test bilinear
        bilinear = Interpolator.interpolate(self.test_image, scale_factor=2.0, method='bilinear')
        self.assertEqual(bilinear.shape, (200, 200, 3))
        self.assertEqual(bilinear.dtype, np.uint8)
        
        # Test bicubic
        bicubic = Interpolator.interpolate(self.test_image, scale_factor=2.0, method='bicubic')
        self.assertEqual(bicubic.shape, (200, 200, 3))
        self.assertEqual(bicubic.dtype, np.uint8)
        
        # Test lanczos
        lanczos = Interpolator.interpolate(self.test_image, scale_factor=2.0, method='lanczos')
        self.assertEqual(lanczos.shape, (200, 200, 3))
        self.assertEqual(lanczos.dtype, np.uint8)
        
        # Test downscaling
        downscaled = Interpolator.interpolate(self.test_image, scale_factor=0.5, method='bicubic')
        self.assertEqual(downscaled.shape, (50, 50, 3))
        self.assertEqual(downscaled.dtype, np.uint8)
        
        # Test with grayscale image
        gray_bicubic = Interpolator.interpolate(self.gray_image, scale_factor=2.0, method='bicubic')
        self.assertEqual(gray_bicubic.shape, (200, 200))
        self.assertEqual(gray_bicubic.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            Interpolator.interpolate(self.test_image, method='invalid')

    def test_false_color(self):
        """Test false color enhancement methods."""
        # Test jet colormap
        jet = FalseColorEnhancer.enhance(self.test_image, method='jet')
        self.assertEqual(jet.shape, self.test_image.shape)
        self.assertEqual(jet.dtype, np.uint8)
        
        # Test hot colormap
        hot = FalseColorEnhancer.enhance(self.test_image, method='hot')
        self.assertEqual(hot.shape, self.test_image.shape)
        self.assertEqual(hot.dtype, np.uint8)
        
        # Test with grayscale image
        gray_jet = FalseColorEnhancer.enhance(self.gray_image, method='jet')
        self.assertEqual(gray_jet.shape, (100, 100, 3))  # Should be converted to BGR
        self.assertEqual(gray_jet.dtype, np.uint8)
        
        # Test with min/max values
        normalized = FalseColorEnhancer.enhance(self.test_image, method='jet', min_value=0, max_value=255)
        self.assertEqual(normalized.shape, self.test_image.shape)
        self.assertEqual(normalized.dtype, np.uint8)
        
        # Test invalid method
        with self.assertRaises(ValueError):
            FalseColorEnhancer.enhance(self.test_image, method='invalid')

    def test_image_metrics(self):
        """Test image metrics calculation."""
        # Test with identical images
        metrics = ImageMetrics.calculate_metrics(self.test_image, self.test_image)
        self.assertIn('mse', metrics)
        self.assertIn('psnr', metrics)
        self.assertIn('entropy', metrics)
        self.assertIn('contrast', metrics)
        self.assertEqual(metrics['mse'], 0)
        self.assertEqual(metrics['psnr'], float('inf'))
        
        # Test with different images
        metrics = ImageMetrics.calculate_metrics(self.test_image, self.noisy_image)
        self.assertIn('mse', metrics)
        self.assertIn('psnr', metrics)
        self.assertIn('entropy', metrics)
        self.assertIn('contrast', metrics)
        self.assertGreater(metrics['mse'], 0)
        self.assertLess(metrics['psnr'], float('inf'))
        
        # Test with grayscale and color images
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        metrics = ImageMetrics.calculate_metrics(self.test_image, gray)
        self.assertIn('mse', metrics)
        self.assertIn('entropy', metrics)
        self.assertIn('contrast', metrics)

    def test_process_image(self):
        """Test the main process_image function."""
        # Test grayscale operation
        processed, metrics = ImageProcessor.process_image(self.test_image, "grayscale")
        self.assertEqual(len(processed.shape), 2)  # Should be grayscale
        self.assertIn('processing_time', metrics)
        
        # Test contrast adjustment operation
        processed, metrics = ImageProcessor.process_image(self.test_image, "contrast")
        self.assertEqual(processed.shape, self.test_image.shape)
        self.assertIn('processing_time', metrics)
        
        # Test noise reduction operation
        processed, metrics = ImageProcessor.process_image(self.noisy_image, "noise_reduction")
        self.assertEqual(processed.shape, self.noisy_image.shape)
        self.assertIn('processing_time', metrics)
        
        # Test edge enhancement operation
        processed, metrics = ImageProcessor.process_image(self.test_image, "edge_enhancement")
        self.assertEqual(processed.shape, self.test_image.shape)
        self.assertIn('processing_time', metrics)
        
        # Test sharpening operation
        processed, metrics = ImageProcessor.process_image(self.blurry_image, "sharpening")
        self.assertEqual(processed.shape, self.blurry_image.shape)
        self.assertIn('processing_time', metrics)
        
        # Test interpolation operation
        processed, metrics = ImageProcessor.process_image(self.test_image, "interpolation")
        self.assertEqual(processed.shape, (200, 200, 3))  # Default scale factor is 2.0
        self.assertIn('processing_time', metrics)
        
        # Test false color operation
        processed, metrics = ImageProcessor.process_image(self.test_image, "false_color")
        self.assertEqual(processed.shape, self.test_image.shape)
        self.assertIn('processing_time', metrics)
        
        # Test with custom parameters
        processed, metrics = ImageProcessor.process_image(
            self.test_image, "grayscale", {"method": "luminosity"}
        )
        self.assertEqual(len(processed.shape), 2)
        self.assertIn('processing_time', metrics)
        
        # Test invalid operation
        with self.assertRaises(ValueError):
            ImageProcessor.process_image(self.test_image, "invalid_operation")


if __name__ == '__main__':
    unittest.main()
