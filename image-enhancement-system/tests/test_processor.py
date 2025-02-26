"""
Tests for the image processor module.
"""

import os
import sys
import unittest
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.processor import ImageProcessor

class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ImageProcessor()
    
    def test_get_image_info(self):
        """Test the get_image_info method."""
        # Create a simple test image
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Get image info
        info = self.processor.get_image_info(test_image)
        
        # Check the results
        self.assertEqual(info['width'], 200)
        self.assertEqual(info['height'], 100)
        self.assertEqual(info['channels'], 3)
        self.assertEqual(info['size'], 100 * 200 * 3)
        self.assertEqual(info['dtype'], 'uint8')

if __name__ == '__main__':
    unittest.main()
