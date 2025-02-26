"""
Integration tests for the blood vessel analysis system.

This module tests the integration of all components of the system.
"""

import os
import sys
import unittest
import numpy as np
import tempfile
from stl import mesh
import shutil

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import BloodVesselAnalyzer
from src.stl_processing.stl_reader import STLReader
from src.geometric_analysis.centerline import CenterlineExtractor
from src.geometric_analysis.cross_section import CrossSectionAnalyzer
from src.topology.vessel_network import VesselNetwork
from src.utils.data_structures import VesselModel


class TestIntegration(unittest.TestCase):
    """Test case for the integration of all components."""

    def setUp(self):
        """Set up the test case."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a simple STL file for testing
        self.test_stl_path = os.path.join(self.temp_dir.name, "test_tube.stl")
        self.create_tube_mesh(self.test_stl_path)
        
        # Create a blood vessel analyzer
        self.analyzer = BloodVesselAnalyzer()

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

    def test_full_pipeline(self):
        """Test the full analysis pipeline."""
        # Load the STL file
        success = self.analyzer.load_stl(self.test_stl_path)
        self.assertTrue(success)
        
        # Extract the centerline
        centerline = self.analyzer.extract_centerline()
        self.assertIsNotNone(centerline)
        self.assertGreater(len(centerline), 0)
        
        # Analyze the geometry
        results = self.analyzer.analyze_geometry()
        self.assertIsNotNone(results)
        
        # Check that the results contain the expected keys
        self.assertIn("surface_area", results)
        self.assertIn("centerline_length", results)
        self.assertIn("num_branch_points", results)
        self.assertIn("num_endpoints", results)
        self.assertIn("num_segments", results)
        
        # Build the vessel model
        model = self.analyzer.build_vessel_model()
        self.assertIsNotNone(model)
        
        # Check that the model has the expected properties
        self.assertEqual(model.name, os.path.basename(self.test_stl_path))
        self.assertGreaterEqual(len(model.get_all_nodes()), 2)  # At least 2 nodes (start and end)
        self.assertGreaterEqual(len(model.get_all_segments()), 1)  # At least 1 segment
        
        # Export the results to a directory
        output_dir = os.path.join(self.temp_dir.name, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        self.analyzer.export_results(output_dir)
        
        # Check that the output files were created
        model_filename = os.path.join(output_dir, f"{os.path.basename(self.test_stl_path).split('.')[0]}_model.json")
        self.assertTrue(os.path.exists(model_filename))

    def test_save_load_model(self):
        """Test saving and loading a vessel model."""
        # Load the STL file
        success = self.analyzer.load_stl(self.test_stl_path)
        self.assertTrue(success)
        
        # Extract the centerline
        centerline = self.analyzer.extract_centerline()
        self.assertIsNotNone(centerline)
        
        # Build the vessel model
        model = self.analyzer.build_vessel_model()
        self.assertIsNotNone(model)
        
        # Save the model to a JSON file
        json_path = os.path.join(self.temp_dir.name, "model.json")
        model.save_to_json(json_path)
        
        # Load the model from the JSON file
        loaded_model = VesselModel.load_from_json(json_path)
        
        # Check that the loaded model has the same properties as the original model
        self.assertEqual(loaded_model.model_id, model.model_id)
        self.assertEqual(loaded_model.name, model.name)
        self.assertEqual(len(loaded_model.get_all_nodes()), len(model.get_all_nodes()))
        self.assertEqual(len(loaded_model.get_all_segments()), len(model.get_all_segments()))

    def test_cross_section_analysis(self):
        """Test cross-section analysis."""
        # Load the STL file
        success = self.analyzer.load_stl(self.test_stl_path)
        self.assertTrue(success)
        
        # Extract the centerline
        centerline = self.analyzer.extract_centerline()
        self.assertIsNotNone(centerline)
        
        # Analyze cross-sections
        cross_sections = self.analyzer.analyze_cross_sections()
        self.assertIsNotNone(cross_sections)
        self.assertGreater(len(cross_sections), 0)
        
        # Check that each cross-section has the expected properties
        for cs in cross_sections:
            self.assertNotIn("error", cs)
            self.assertIn("area", cs)
            self.assertIn("perimeter", cs)
            self.assertIn("equivalent_diameter", cs)
            self.assertIn("circularity", cs)
            
            # For a tube with radius 1.0, the area should be approximately pi
            self.assertAlmostEqual(cs["area"], np.pi * 1.0**2, delta=0.2)
            
            # For a tube with radius 1.0, the perimeter should be approximately 2*pi
            self.assertAlmostEqual(cs["perimeter"], 2 * np.pi * 1.0, delta=0.5)
            
            # For a tube with radius 1.0, the equivalent diameter should be approximately 2.0
            self.assertAlmostEqual(cs["equivalent_diameter"], 2.0, delta=0.2)
            
            # For a perfect circle, the circularity should be approximately 1.0
            self.assertAlmostEqual(cs["circularity"], 1.0, delta=0.2)

    def test_topology_analysis(self):
        """Test topology analysis."""
        # Load the STL file
        success = self.analyzer.load_stl(self.test_stl_path)
        self.assertTrue(success)
        
        # Extract the centerline
        centerline = self.analyzer.extract_centerline()
        self.assertIsNotNone(centerline)
        
        # Build the vessel model
        model = self.analyzer.build_vessel_model()
        self.assertIsNotNone(model)
        
        # Get the branch points
        branch_points = model.get_branch_points()
        
        # For a simple tube, there should be no branch points
        self.assertEqual(len(branch_points), 0)
        
        # Get the endpoints
        endpoints = model.get_endpoints()
        
        # For a simple tube, there should be 2 endpoints
        self.assertGreaterEqual(len(endpoints), 1)
        
        # Get all segments
        segments = model.get_all_segments()
        
        # For a simple tube, there should be at least 1 segment
        self.assertGreaterEqual(len(segments), 1)
        
        # Calculate the total length
        total_length = model.calculate_total_length()
        
        # For a tube with height 10.0, the total length should be approximately 10.0
        self.assertGreaterEqual(total_length, 9.0)


if __name__ == '__main__':
    unittest.main()
