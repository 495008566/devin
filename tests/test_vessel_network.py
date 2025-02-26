"""
Unit tests for the vessel network module.
"""

import os
import sys
import unittest
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.topology.vessel_network import VesselNetwork


class TestVesselNetwork(unittest.TestCase):
    """Test case for the VesselNetwork class."""

    def setUp(self):
        """Set up the test case."""
        self.network = VesselNetwork()
        
        # Create a simple Y-shaped vessel network
        self.centerline_points = np.array([
            [0, 0, 0],  # Start point
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 0, 4],  # Branch point
            [1, 0, 5],  # Branch 1
            [2, 0, 6],
            [3, 0, 7],  # Endpoint 1
            [-1, 0, 5],  # Branch 2
            [-2, 0, 6],
            [-3, 0, 7]   # Endpoint 2
        ])
        
        self.branch_points = np.array([[0, 0, 4]])
        self.endpoints = np.array([[0, 0, 0], [3, 0, 7], [-3, 0, 7]])
        
        self.segments = [
            np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4]]),  # Segment 1
            np.array([[0, 0, 4], [1, 0, 5], [2, 0, 6], [3, 0, 7]]),  # Segment 2
            np.array([[0, 0, 4], [-1, 0, 5], [-2, 0, 6], [-3, 0, 7]])  # Segment 3
        ]
        
        # Build the network
        self.network.build_from_centerline(
            self.centerline_points,
            self.branch_points,
            self.endpoints,
            self.segments
        )

    def test_build_from_centerline(self):
        """Test building a vessel network from centerline data."""
        # Check that the graph was created
        self.assertIsNotNone(self.network.graph)
        
        # Check that the graph has the correct number of nodes
        self.assertEqual(len(self.network.graph.nodes), 4)  # 1 branch point + 3 endpoints
        
        # Check that the graph has the correct number of edges
        self.assertEqual(len(self.network.graph.edges), 3)  # 3 segments

    def test_get_branch_points(self):
        """Test getting branch points."""
        # Get the branch points
        branch_points = self.network.get_branch_points()
        
        # Check that the branch points are correct
        self.assertEqual(len(branch_points), 1)
        np.testing.assert_array_equal(branch_points, self.branch_points)

    def test_get_endpoints(self):
        """Test getting endpoints."""
        # Get the endpoints
        endpoints = self.network.get_endpoints()
        
        # Check that the endpoints are correct
        self.assertEqual(len(endpoints), 3)
        
        # Check that all expected endpoints are in the result
        for endpoint in self.endpoints:
            self.assertTrue(any(np.array_equal(endpoint, ep) for ep in endpoints))

    def test_get_segments(self):
        """Test getting segments."""
        # Get the segments
        segments = self.network.get_segments()
        
        # Check that the segments are correct
        self.assertEqual(len(segments), 3)
        
        # Check that all expected segments are in the result
        for segment in self.segments:
            self.assertTrue(any(np.array_equal(segment, seg) for seg in segments))

    def test_get_segment_lengths(self):
        """Test getting segment lengths."""
        # Get the segment lengths
        lengths = self.network.get_segment_lengths()
        
        # Check that the lengths are correct
        self.assertEqual(len(lengths), 3)
        
        # Check that the lengths are positive
        for length in lengths.values():
            self.assertGreater(length, 0.0)

    def test_get_bifurcation_angles(self):
        """Test getting bifurcation angles."""
        # Get the bifurcation angles
        angles = self.network.get_bifurcation_angles()
        
        # Check that the angles are correct
        self.assertEqual(len(angles), 1)  # 1 branch point
        
        # Check that the branch point has 1 angle (between the 2 branches)
        self.assertEqual(len(list(angles.values())[0]), 1)
        
        # Check that the angle is between 0 and 180 degrees
        angle = list(angles.values())[0][0]
        self.assertGreaterEqual(angle, 0.0)
        self.assertLessEqual(angle, 180.0)

    def test_calculate_tortuosity(self):
        """Test calculating tortuosity."""
        # Calculate tortuosity for each segment
        for segment in self.segments:
            tortuosity = self.network.calculate_tortuosity(segment)
            
            # Check that the tortuosity is at least 1.0
            self.assertGreaterEqual(tortuosity, 1.0)

    def test_calculate_all_tortuosities(self):
        """Test calculating all tortuosities."""
        # Calculate all tortuosities
        tortuosities = self.network.calculate_all_tortuosities()
        
        # Check that the tortuosities are correct
        self.assertEqual(len(tortuosities), 3)  # 3 segments
        
        # Check that the tortuosities are at least 1.0
        for tortuosity in tortuosities.values():
            self.assertGreaterEqual(tortuosity, 1.0)

    def test_get_topological_features(self):
        """Test getting topological features."""
        # Get the topological features
        features = self.network.get_topological_features()
        
        # Check that the features are correct
        self.assertEqual(features["num_nodes"], 4)  # 1 branch point + 3 endpoints
        self.assertEqual(features["num_edges"], 3)  # 3 segments
        self.assertEqual(features["num_branch_points"], 1)
        self.assertEqual(features["num_endpoints"], 3)

    def test_export_to_dict(self):
        """Test exporting the network to a dictionary."""
        # Export the network to a dictionary
        data = self.network.export_to_dict()
        
        # Check that the dictionary contains the expected keys
        self.assertIn("nodes", data)
        self.assertIn("edges", data)
        self.assertIn("features", data)
        self.assertIn("tortuosities", data)
        self.assertIn("bifurcation_angles", data)
        
        # Check that the dictionary contains the correct number of nodes and edges
        self.assertEqual(len(data["nodes"]), 4)  # 1 branch point + 3 endpoints
        self.assertEqual(len(data["edges"]), 3)  # 3 segments


if __name__ == '__main__':
    unittest.main()
