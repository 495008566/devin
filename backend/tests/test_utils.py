import unittest
import numpy as np
import cv2
import base64
import sys
from pathlib import Path

# Add the parent directory to sys.path to import the app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.image_processing.utils import (
    encode_image_to_base64, decode_base64_to_image,
    format_metrics, get_available_operations,
    optimize_for_large_images, restore_original_size
)

class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test image (100x100 with a white square in the middle)
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.test_image[25:75, 25:75] = 255  # White square
        
        # Create a grayscale test image
        self.gray_image = np.zeros((100, 100), dtype=np.uint8)
        self.gray_image[25:75, 25:75] = 255  # White square
        
        # Create a large test image
        self.large_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        self.large_image[500:1500, 500:1500] = 255  # White square

    def test_base64_conversion(self):
        """Test base64 encoding and decoding."""
        # Test encoding color image
        base64_str = encode_image_to_base64(self.test_image)
        self.assertTrue(base64_str.startswith('data:image/jpeg;base64,'))
        
        # Test encoding grayscale image
        gray_base64 = encode_image_to_base64(self.gray_image)
        self.assertTrue(gray_base64.startswith('data:image/jpeg;base64,'))
        
        # Test decoding
        decoded = decode_base64_to_image(base64_str)
        self.assertEqual(decoded.shape, self.test_image.shape)
        
        # Test decoding without data URL prefix
        base64_content = base64_str.split(',', 1)[1]
        decoded_no_prefix = decode_base64_to_image(base64_content)
        self.assertEqual(decoded_no_prefix.shape, self.test_image.shape)
        
        # Test invalid base64 string
        with self.assertRaises(Exception):
            decode_base64_to_image("invalid_base64_string")

    def test_format_metrics(self):
        """Test metrics formatting."""
        metrics = {
            'mse': 10.5,
            'psnr': 28.7,
            'entropy': 5.2,
            'contrast': 45.8,
            'processing_time': 0.123
        }
        
        formatted = format_metrics(metrics)
        self.assertEqual(formatted['mse'], '10.50')
        self.assertEqual(formatted['psnr'], '28.70 dB')
        self.assertEqual(formatted['entropy'], '5.20 bits')
        self.assertEqual(formatted['contrast'], '45.80')
        self.assertEqual(formatted['processing_time'], '0.123 seconds')
        
        # Test infinite PSNR
        inf_metrics = {'psnr': float('inf')}
        formatted_inf = format_metrics(inf_metrics)
        self.assertEqual(formatted_inf['psnr'], '∞ dB')

    def test_get_available_operations(self):
        """Test getting available operations."""
        operations = get_available_operations()
        self.assertIsInstance(operations, list)
        self.assertGreater(len(operations), 0)
        
        # Check structure of operations
        for op in operations:
            self.assertIn('id', op)
            self.assertIn('name', op)
            self.assertIn('description', op)
            self.assertIn('parameters', op)
            
            # Check parameters structure
            for param in op['parameters']:
                self.assertIn('name', param)
                self.assertIn('type', param)
                self.assertIn('description', param)
                
                # Check select type parameters
                if param['type'] == 'select':
                    self.assertIn('options', param)
                    self.assertIn('default', param)
                    self.assertIn(param['default'], param['options'])
                
                # Check range type parameters
                if param['type'] == 'range':
                    self.assertIn('min', param)
                    self.assertIn('max', param)
                    self.assertIn('step', param)
                    self.assertIn('default', param)
                    self.assertGreaterEqual(param['default'], param['min'])
                    self.assertLessEqual(param['default'], param['max'])

    def test_optimize_for_large_images(self):
        """Test image optimization for large images."""
        # Test with small image (should not resize)
        optimized, scale = optimize_for_large_images(self.test_image, max_size=1024)
        self.assertEqual(optimized.shape, self.test_image.shape)
        self.assertEqual(scale, 1.0)
        
        # Test with large image (should resize)
        optimized, scale = optimize_for_large_images(self.large_image, max_size=1024)
        self.assertEqual(optimized.shape, (1024, 1024, 3))
        self.assertAlmostEqual(scale, 1024/2000, places=3)
        
        # Test with custom max size
        optimized, scale = optimize_for_large_images(self.large_image, max_size=500)
        self.assertEqual(optimized.shape, (500, 500, 3))
        self.assertAlmostEqual(scale, 500/2000, places=3)
        
        # Test with grayscale image
        gray_optimized, gray_scale = optimize_for_large_images(
            np.zeros((2000, 1000), dtype=np.uint8), max_size=1000
        )
        self.assertEqual(gray_optimized.shape, (1000, 500))
        self.assertAlmostEqual(gray_scale, 0.5, places=3)

    def test_restore_original_size(self):
        """Test restoring image to original size."""
        # Resize image
        small = cv2.resize(self.test_image, (50, 50))
        
        # Restore to original size
        restored = restore_original_size(small, self.test_image.shape[:2])
        self.assertEqual(restored.shape, self.test_image.shape)
        
        # Test with different interpolation method
        restored_lanczos = restore_original_size(small, self.test_image.shape[:2], 
                                               interpolation=cv2.INTER_LANCZOS4)
        self.assertEqual(restored_lanczos.shape, self.test_image.shape)
        
        # Test with grayscale image
        gray_small = cv2.resize(self.gray_image, (50, 50))
        gray_restored = restore_original_size(gray_small, self.gray_image.shape[:2])
        self.assertEqual(gray_restored.shape, self.gray_image.shape)


if __name__ == '__main__':
    unittest.main()
