"""
Unit tests for the STL reader module.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
from stl import mesh

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stl_processing.stl_reader import STLReader


class TestSTLReader(unittest.TestCase):
    """Test case for the STLReader class."""

    def setUp(self):
        """Set up the test case."""
        self.reader = STLReader()
        
        # Create a simple STL file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_stl_path = os.path.join(self.temp_dir.name, "test.stl")
        
        # Create a simple cube mesh
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ])
        
        # Define the 12 triangles composing the cube
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
            [4, 7, 6],
            [4, 6, 5]
        ])
        
        # Create the mesh
        cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                cube.vectors[i][j] = vertices[f[j], :]
        
        # Save the mesh to the temporary file
        cube.save(self.test_stl_path)

    def tearDown(self):
        """Tear down the test case."""
        self.temp_dir.cleanup()

    def test_read_file(self):
        """Test reading an STL file."""
        # Test reading a valid STL file
        success = self.reader.read_file(self.test_stl_path)
        self.assertTrue(success)
        self.assertIsNotNone(self.reader.mesh)
        
        # Test reading a non-existent file
        success = self.reader.read_file("non_existent.stl")
        self.assertFalse(success)

    def test_get_mesh_info(self):
        """Test getting mesh information."""
        # Read the test STL file
        self.reader.read_file(self.test_stl_path)
        
        # Get the mesh info
        info = self.reader.get_mesh_info()
        
        # Check that the info contains the expected keys
        self.assertIn("type", info)
        self.assertIn("num_triangles", info)
        
        # Check that the number of triangles is correct
        self.assertEqual(info["num_triangles"], 12)

    def test_get_bounding_box(self):
        """Test getting the bounding box of the mesh."""
        # Read the test STL file
        self.reader.read_file(self.test_stl_path)
        
        # Get the bounding box
        min_point, max_point = self.reader.get_bounding_box()
        
        # Check that the bounding box is correct
        np.testing.assert_array_almost_equal(min_point, [0, 0, 0])
        np.testing.assert_array_almost_equal(max_point, [1, 1, 1])

    def test_get_surface_area(self):
        """Test calculating the surface area of the mesh."""
        # Read the test STL file
        self.reader.read_file(self.test_stl_path)
        
        # Get the surface area
        area = self.reader.get_surface_area()
        
        # Check that the area is correct (6 faces of a unit cube)
        self.assertAlmostEqual(area, 6.0, places=5)

    def test_validate_mesh(self):
        """Test validating the mesh."""
        # Read the test STL file
        self.reader.read_file(self.test_stl_path)
        
        # Validate the mesh
        validation = self.reader.validate_mesh()
        
        # Check that the validation result is correct
        self.assertIn("valid", validation)
        self.assertTrue(validation["valid"])


if __name__ == '__main__':
    unittest.main()
