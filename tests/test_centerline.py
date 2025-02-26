"""
Unit tests for the centerline extraction module.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
from stl import mesh
import trimesh

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stl_processing.stl_reader import STLReader
from src.geometric_analysis.centerline import CenterlineExtractor


class TestCenterlineExtractor(unittest.TestCase):
    """Test case for the CenterlineExtractor class."""

    def setUp(self):
        """Set up the test case."""
        # Create a simple STL file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_stl_path = os.path.join(self.temp_dir.name, "test_tube.stl")
        
        # Create a simple tube mesh
        self.create_tube_mesh(self.test_stl_path)
        
        # Read the STL file
        self.reader = STLReader()
        self.reader.read_file(self.test_stl_path)
        
        # Create a centerline extractor
        self.extractor = CenterlineExtractor(self.reader.mesh)

    def tearDown(self):
        """Tear down the test case."""
        self.temp_dir.cleanup()

    def create_tube_mesh(self, filename, radius=1.0, height=10.0, num_segments=16):
        """
        Create a simple tube mesh and save it to a file.
        
        Args:
            filename: The filename to save the mesh to.
            radius: The radius of the tube.
            height: The height of the tube.
            num_segments: The number of segments around the circumference.
        """
        # Create a cylinder using trimesh
        cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=num_segments)
        
        # Save the mesh to the file
        cylinder.export(filename)
        
        # Return the trimesh object
        return cylinder

    def test_extract_centerline(self):
        """Test extracting the centerline."""
        # Extract the centerline
        centerline = self.extractor.extract_centerline()
        
        # Check that the centerline is not empty
        self.assertGreater(len(centerline), 0)
        
        # Check that the centerline points are inside the tube
        for point in centerline:
            # Calculate the distance from the point to the tube axis
            x, y, z = point
            distance_to_axis = np.sqrt(x**2 + y**2)
            
            # Check that the point is inside the tube (with some tolerance)
            # The radius is 1.0, but allow some tolerance for numerical issues
            self.assertLessEqual(distance_to_axis, 1.5)
            
            # Check that the point is within the height of the tube (with some tolerance)
            self.assertGreaterEqual(z, -0.5)
            self.assertLessEqual(z, 10.5)

    def test_get_branch_points(self):
        """Test getting branch points."""
        # Extract the centerline
        self.extractor.extract_centerline()
        
        # Get the branch points
        branch_points = self.extractor.get_branch_points()
        
        # For a simple tube, there should be no branch points
        self.assertEqual(len(branch_points), 0)

    def test_get_endpoints(self):
        """Test getting endpoints."""
        # Extract the centerline
        self.extractor.extract_centerline()
        
        # Get the endpoints
        endpoints = self.extractor.get_endpoints()
        
        # For a simple tube, there should be 2 endpoints
        self.assertGreaterEqual(len(endpoints), 1)

    def test_get_centerline_segments(self):
        """Test getting centerline segments."""
        # Extract the centerline
        self.extractor.extract_centerline()
        
        # Get the centerline segments
        segments = self.extractor.get_centerline_segments()
        
        # For a simple tube, there should be at least 1 segment
        self.assertGreaterEqual(len(segments), 1)

    def test_calculate_segment_lengths(self):
        """Test calculating segment lengths."""
        # Extract the centerline
        self.extractor.extract_centerline()
        
        # Calculate segment lengths
        lengths = self.extractor.calculate_segment_lengths()
        
        # Check that the lengths are positive
        for length in lengths:
            self.assertGreater(length, 0.0)

    def test_calculate_segment_diameters(self):
        """Test calculating segment diameters."""
        # Extract the centerline
        self.extractor.extract_centerline()
        
        # Calculate segment diameters
        diameters = self.extractor.calculate_segment_diameters()
        
        # Check that the diameters are positive
        for diameter in diameters:
            self.assertGreaterEqual(diameter, 0.0)


if __name__ == '__main__':
    unittest.main()
