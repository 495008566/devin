"""
Tests for the database manager module.
"""

import os
import sys
import unittest
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.db_manager import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    """Test cases for the DatabaseManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Use in-memory SQLite database for testing
        self.db = DatabaseManager(use_in_memory=True)
    
    def test_add_image(self):
        """Test adding an image to the database."""
        image_data = {
            'filename': 'test.jpg',
            'file_size': 1024,
            'width': 800,
            'height': 600,
            'color_space': 'BGR',
            'file_format': 'JPEG'
        }
        
        image_id = self.db.add_image(image_data)
        self.assertIsNotNone(image_id)
        
        # Verify the image was added
        image = self.db.get_image(image_id)
        self.assertIsNotNone(image)
        self.assertEqual(image['filename'], 'test.jpg')
        self.assertEqual(image['width'], 800)
        self.assertEqual(image['height'], 600)
    
    def test_add_enhancement(self):
        """Test adding an enhancement to the database."""
        # First add an image
        image_data = {
            'filename': 'test.jpg',
            'file_size': 1024,
            'width': 800,
            'height': 600,
            'color_space': 'BGR',
            'file_format': 'JPEG'
        }
        
        image_id = self.db.add_image(image_data)
        
        # Add an enhancement
        enhancement_data = {
            'image_id': image_id,
            'enhancement_type': 'grayscale',
            'parameters': {'method': 'weighted'},
            'output_filename': 'test_gray.jpg'
        }
        
        enhancement_id = self.db.add_enhancement(enhancement_data)
        self.assertIsNotNone(enhancement_id)
        
        # Verify the enhancement was added
        enhancements = self.db.get_enhancements_for_image(image_id)
        self.assertEqual(len(enhancements), 1)
        self.assertEqual(enhancements[0]['enhancement_type'], 'grayscale')
        self.assertEqual(enhancements[0]['output_filename'], 'test_gray.jpg')
    
    def test_add_metrics(self):
        """Test adding metrics to the database."""
        # First add an image
        image_id = self.db.add_image({
            'filename': 'test.jpg',
            'file_size': 1024,
            'width': 800,
            'height': 600
        })
        
        # Add an enhancement
        enhancement_id = self.db.add_enhancement({
            'image_id': image_id,
            'enhancement_type': 'contrast',
            'parameters': {'alpha': 1.5, 'beta': 10},
            'output_filename': 'test_contrast.jpg'
        })
        
        # Add metrics
        metrics_data = [
            {
                'enhancement_id': enhancement_id,
                'metric_name': 'psnr',
                'metric_value': 32.5
            },
            {
                'enhancement_id': enhancement_id,
                'metric_name': 'ssim',
                'metric_value': 0.92
            }
        ]
        
        result = self.db.add_metrics(metrics_data)
        self.assertTrue(result)
        
        # Verify the metrics were added
        metrics = self.db.get_metrics_for_enhancement(enhancement_id)
        self.assertEqual(len(metrics), 2)
        
        # Check metric values
        metric_values = {m['metric_name']: m['metric_value'] for m in metrics}
        self.assertAlmostEqual(metric_values['psnr'], 32.5)
        self.assertAlmostEqual(metric_values['ssim'], 0.92)

if __name__ == '__main__':
    unittest.main()
