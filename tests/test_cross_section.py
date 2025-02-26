"""
Unit tests for the cross-section analysis module.
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
from src.geometric_analysis.cross_section import CrossSectionAnalyzer


class TestCrossSectionAnalyzer(unittest.TestCase):
    """Test case for the CrossSectionAnalyzer class."""

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
        
        # Create a centerline for the tube
        self.centerline = np.array([
            [0, 0, 0],
            [0, 0, 2.5],
            [0, 0, 5.0],
            [0, 0, 7.5],
            [0, 0, 10.0]
        ])
        
        # Create a cross-section analyzer
        self.analyzer = CrossSectionAnalyzer(self.reader.mesh)

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
        # Create vertices for the tube
        vertices = []
        
        # Create the bottom and top circles
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            # Bottom circle
            vertices.append([x, y, 0])
            
            # Top circle
            vertices.append([x, y, height])
        
        vertices = np.array(vertices)
        
        # Create faces for the tube
        faces = []
        
        # Create the triangles for the tube walls
        for i in range(num_segments):
            # Get the indices of the vertices for this segment
            i1 = 2 * i
            i2 = 2 * i + 1
            i3 = 2 * ((i + 1) % num_segments)
            i4 = 2 * ((i + 1) % num_segments) + 1
            
            # Add the two triangles for this segment
            faces.append([i1, i2, i3])
            faces.append([i2, i4, i3])
        
        faces = np.array(faces)
        
        # Create the mesh
        tube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
        for i, f in enumerate(faces):
            for j in range(3):
                tube.vectors[i][j] = vertices[f[j], :]
        
        # Save the mesh to the file
        tube.save(filename)

    def test_compute_cross_section(self):
        """Test computing a cross-section."""
        # Compute a cross-section at the middle of the tube
        point = np.array([0, 0, 5.0])
        normal = np.array([0, 0, 1.0])
        
        cross_section = self.analyzer.compute_cross_section(point, normal)
        
        # Check that the cross-section was computed successfully
        self.assertNotIn("error", cross_section)
        
        # Check that the cross-section has the expected properties
        self.assertIn("vertices_2d", cross_section)
        self.assertIn("vertices_3d", cross_section)
        self.assertIn("area", cross_section)
        self.assertIn("perimeter", cross_section)
        self.assertIn("equivalent_diameter", cross_section)
        self.assertIn("circularity", cross_section)
        
        # Check that the area is approximately pi * r^2
        self.assertAlmostEqual(cross_section["area"], np.pi * 1.0**2, delta=0.2)
        
        # Check that the perimeter is approximately 2 * pi * r
        self.assertAlmostEqual(cross_section["perimeter"], 2 * np.pi * 1.0, delta=0.5)
        
        # Check that the equivalent diameter is approximately 2 * r
        self.assertAlmostEqual(cross_section["equivalent_diameter"], 2 * 1.0, delta=0.2)
        
        # Check that the circularity is approximately 1.0 (perfect circle)
        self.assertAlmostEqual(cross_section["circularity"], 1.0, delta=0.2)

    def test_compute_cross_sections_along_centerline(self):
        """Test computing cross-sections along a centerline."""
        # Compute cross-sections along the centerline
        cross_sections = self.analyzer.compute_cross_sections_along_centerline(self.centerline)
        
        # Check that cross-sections were computed successfully
        self.assertEqual(len(cross_sections), len(self.centerline))
        
        # Check that each cross-section has the expected properties
        for cs in cross_sections:
            self.assertNotIn("error", cs)
            self.assertIn("vertices_2d", cs)
            self.assertIn("vertices_3d", cs)
            self.assertIn("area", cs)
            self.assertIn("perimeter", cs)
            self.assertIn("equivalent_diameter", cs)
            self.assertIn("circularity", cs)
            
            # Check that the area is approximately pi * r^2
            self.assertAlmostEqual(cs["area"], np.pi * 1.0**2, delta=0.2)
            
            # Check that the perimeter is approximately 2 * pi * r
            self.assertAlmostEqual(cs["perimeter"], 2 * np.pi * 1.0, delta=0.5)
            
            # Check that the equivalent diameter is approximately 2 * r
            self.assertAlmostEqual(cs["equivalent_diameter"], 2 * 1.0, delta=0.2)
            
            # Check that the circularity is approximately 1.0 (perfect circle)
            self.assertAlmostEqual(cs["circularity"], 1.0, delta=0.2)

    def test_compute_cross_sections_at_intervals(self):
        """Test computing cross-sections at intervals."""
        # Compute cross-sections at intervals
        cross_sections = self.analyzer.compute_cross_sections_at_intervals(self.centerline, interval=2.5)
        
        # Check that cross-sections were computed successfully
        self.assertGreaterEqual(len(cross_sections), 4)  # At least 4 cross-sections for a 10-unit tube with 2.5-unit intervals
        
        # Check that each cross-section has the expected properties
        for cs in cross_sections:
            self.assertNotIn("error", cs)
            self.assertIn("vertices_2d", cs)
            self.assertIn("vertices_3d", cs)
            self.assertIn("area", cs)
            self.assertIn("perimeter", cs)
            self.assertIn("equivalent_diameter", cs)
            self.assertIn("circularity", cs)
            
            # Check that the area is approximately pi * r^2
            self.assertAlmostEqual(cs["area"], np.pi * 1.0**2, delta=0.2)
            
            # Check that the perimeter is approximately 2 * pi * r
            self.assertAlmostEqual(cs["perimeter"], 2 * np.pi * 1.0, delta=0.5)
            
            # Check that the equivalent diameter is approximately 2 * r
            self.assertAlmostEqual(cs["equivalent_diameter"], 2 * 1.0, delta=0.2)
            
            # Check that the circularity is approximately 1.0 (perfect circle)
            self.assertAlmostEqual(cs["circularity"], 1.0, delta=0.2)


if __name__ == '__main__':
    unittest.main()
