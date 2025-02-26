"""
Unit tests for the data structures module.
"""

import os
import sys
import unittest
import numpy as np
import json
import tempfile

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_structures import VesselNode, VesselSegment, VesselModel


class TestVesselNode(unittest.TestCase):
    """Test case for the VesselNode class."""

    def setUp(self):
        """Set up the test case."""
        self.position = np.array([1.0, 2.0, 3.0])
        self.node = VesselNode(node_id="node1", position=self.position, node_type="branch")

    def test_initialization(self):
        """Test initialization of a vessel node."""
        self.assertEqual(self.node.node_id, "node1")
        np.testing.assert_array_equal(self.node.position, self.position)
        self.assertEqual(self.node.node_type, "branch")
        self.assertEqual(len(self.node.properties), 0)

    def test_set_get_property(self):
        """Test setting and getting properties."""
        # Set a property
        self.node.set_property("test_key", "test_value")
        
        # Get the property
        value = self.node.get_property("test_key")
        
        # Check that the property was set correctly
        self.assertEqual(value, "test_value")
        
        # Get a non-existent property with a default value
        value = self.node.get_property("non_existent", "default")
        
        # Check that the default value was returned
        self.assertEqual(value, "default")

    def test_to_dict(self):
        """Test converting a node to a dictionary."""
        # Set some properties
        self.node.set_property("test_key", "test_value")
        self.node.set_property("test_array", np.array([1, 2, 3]))
        
        # Convert to a dictionary
        node_dict = self.node.to_dict()
        
        # Check that the dictionary contains the expected keys
        self.assertIn("node_id", node_dict)
        self.assertIn("position", node_dict)
        self.assertIn("node_type", node_dict)
        self.assertIn("properties", node_dict)
        
        # Check that the values are correct
        self.assertEqual(node_dict["node_id"], "node1")
        np.testing.assert_array_equal(node_dict["position"], self.position)
        self.assertEqual(node_dict["node_type"], "branch")
        self.assertEqual(node_dict["properties"]["test_key"], "test_value")
        np.testing.assert_array_equal(node_dict["properties"]["test_array"], np.array([1, 2, 3]))

    def test_from_dict(self):
        """Test creating a node from a dictionary."""
        # Create a dictionary
        node_dict = {
            "node_id": "node2",
            "position": [4.0, 5.0, 6.0],
            "node_type": "endpoint",
            "properties": {
                "test_key": "test_value",
                "test_array": [1, 2, 3]
            }
        }
        
        # Create a node from the dictionary
        node = VesselNode.from_dict(node_dict)
        
        # Check that the node was created correctly
        self.assertEqual(node.node_id, "node2")
        np.testing.assert_array_equal(node.position, np.array([4.0, 5.0, 6.0]))
        self.assertEqual(node.node_type, "endpoint")
        self.assertEqual(node.get_property("test_key"), "test_value")
        np.testing.assert_array_equal(node.get_property("test_array"), np.array([1, 2, 3]))


class TestVesselSegment(unittest.TestCase):
    """Test case for the VesselSegment class."""

    def setUp(self):
        """Set up the test case."""
        self.points = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ])
        self.segment = VesselSegment(
            segment_id=1,
            points=self.points,
            start_node_id="node1",
            end_node_id="node2"
        )

    def test_initialization(self):
        """Test initialization of a vessel segment."""
        self.assertEqual(self.segment.segment_id, 1)
        np.testing.assert_array_equal(self.segment.points, self.points)
        self.assertEqual(self.segment.start_node_id, "node1")
        self.assertEqual(self.segment.end_node_id, "node2")
        self.assertEqual(len(self.segment.properties), 0)

    def test_set_get_property(self):
        """Test setting and getting properties."""
        # Set a property
        self.segment.set_property("test_key", "test_value")
        
        # Get the property
        value = self.segment.get_property("test_key")
        
        # Check that the property was set correctly
        self.assertEqual(value, "test_value")
        
        # Get a non-existent property with a default value
        value = self.segment.get_property("non_existent", "default")
        
        # Check that the default value was returned
        self.assertEqual(value, "default")

    def test_calculate_length(self):
        """Test calculating the length of a segment."""
        # Calculate the length
        length = self.segment.calculate_length()
        
        # Check that the length is correct
        self.assertEqual(length, 3.0)

    def test_get_point_at_distance(self):
        """Test getting a point at a specific distance along a segment."""
        # Get a point at distance 1.5
        point = self.segment.get_point_at_distance(1.5)
        
        # Check that the point is correct
        np.testing.assert_array_equal(point, np.array([1.5, 0.0, 0.0]))
        
        # Get a point at distance 0
        point = self.segment.get_point_at_distance(0.0)
        
        # Check that the point is correct
        np.testing.assert_array_equal(point, np.array([0.0, 0.0, 0.0]))
        
        # Get a point at distance equal to the length
        point = self.segment.get_point_at_distance(3.0)
        
        # Check that the point is correct
        np.testing.assert_array_equal(point, np.array([3.0, 0.0, 0.0]))
        
        # Get a point at distance greater than the length
        point = self.segment.get_point_at_distance(4.0)
        
        # Check that the point is correct (should be the last point)
        np.testing.assert_array_equal(point, np.array([3.0, 0.0, 0.0]))

    def test_get_direction_at_distance(self):
        """Test getting the direction at a specific distance along a segment."""
        # Get the direction at distance 1.5
        direction = self.segment.get_direction_at_distance(1.5)
        
        # Check that the direction is correct
        np.testing.assert_array_equal(direction, np.array([1.0, 0.0, 0.0]))

    def test_to_dict(self):
        """Test converting a segment to a dictionary."""
        # Set some properties
        self.segment.set_property("test_key", "test_value")
        self.segment.set_property("test_array", np.array([1, 2, 3]))
        
        # Convert to a dictionary
        segment_dict = self.segment.to_dict()
        
        # Check that the dictionary contains the expected keys
        self.assertIn("segment_id", segment_dict)
        self.assertIn("points", segment_dict)
        self.assertIn("start_node_id", segment_dict)
        self.assertIn("end_node_id", segment_dict)
        self.assertIn("properties", segment_dict)
        
        # Check that the values are correct
        self.assertEqual(segment_dict["segment_id"], 1)
        np.testing.assert_array_equal(segment_dict["points"], self.points)
        self.assertEqual(segment_dict["start_node_id"], "node1")
        self.assertEqual(segment_dict["end_node_id"], "node2")
        self.assertEqual(segment_dict["properties"]["test_key"], "test_value")
        np.testing.assert_array_equal(segment_dict["properties"]["test_array"], np.array([1, 2, 3]))

    def test_from_dict(self):
        """Test creating a segment from a dictionary."""
        # Create a dictionary
        segment_dict = {
            "segment_id": 2,
            "points": [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0]
            ],
            "start_node_id": "node3",
            "end_node_id": "node4",
            "properties": {
                "test_key": "test_value",
                "test_array": [1, 2, 3]
            }
        }
        
        # Create a segment from the dictionary
        segment = VesselSegment.from_dict(segment_dict)
        
        # Check that the segment was created correctly
        self.assertEqual(segment.segment_id, 2)
        np.testing.assert_array_equal(segment.points, np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0]
        ]))
        self.assertEqual(segment.start_node_id, "node3")
        self.assertEqual(segment.end_node_id, "node4")
        self.assertEqual(segment.get_property("test_key"), "test_value")
        np.testing.assert_array_equal(segment.get_property("test_array"), np.array([1, 2, 3]))


class TestVesselModel(unittest.TestCase):
    """Test case for the VesselModel class."""

    def setUp(self):
        """Set up the test case."""
        self.model = VesselModel(model_id="model1", name="Test Model")
        
        # Create some nodes
        self.node1 = VesselNode(node_id="node1", position=np.array([0.0, 0.0, 0.0]), node_type="endpoint")
        self.node2 = VesselNode(node_id="node2", position=np.array([3.0, 0.0, 0.0]), node_type="branch")
        self.node3 = VesselNode(node_id="node3", position=np.array([3.0, 3.0, 0.0]), node_type="endpoint")
        self.node4 = VesselNode(node_id="node4", position=np.array([3.0, -3.0, 0.0]), node_type="endpoint")
        
        # Create some segments
        self.segment1 = VesselSegment(
            segment_id=1,
            points=np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0]
            ]),
            start_node_id="node1",
            end_node_id="node2"
        )
        
        self.segment2 = VesselSegment(
            segment_id=2,
            points=np.array([
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 2.0, 0.0],
                [3.0, 3.0, 0.0]
            ]),
            start_node_id="node2",
            end_node_id="node3"
        )
        
        self.segment3 = VesselSegment(
            segment_id=3,
            points=np.array([
                [3.0, 0.0, 0.0],
                [3.0, -1.0, 0.0],
                [3.0, -2.0, 0.0],
                [3.0, -3.0, 0.0]
            ]),
            start_node_id="node2",
            end_node_id="node4"
        )
        
        # Add the nodes and segments to the model
        self.model.add_node(self.node1)
        self.model.add_node(self.node2)
        self.model.add_node(self.node3)
        self.model.add_node(self.node4)
        
        self.model.add_segment(self.segment1)
        self.model.add_segment(self.segment2)
        self.model.add_segment(self.segment3)

    def test_initialization(self):
        """Test initialization of a vessel model."""
        model = VesselModel(model_id="model2", name="Test Model 2")
        self.assertEqual(model.model_id, "model2")
        self.assertEqual(model.name, "Test Model 2")
        self.assertEqual(len(model.get_all_nodes()), 0)
        self.assertEqual(len(model.get_all_segments()), 0)
        self.assertEqual(len(model.properties), 0)

    def test_add_get_node(self):
        """Test adding and getting nodes."""
        # Get a node
        node = self.model.get_node("node1")
        
        # Check that the node is correct
        self.assertEqual(node.node_id, "node1")
        np.testing.assert_array_equal(node.position, np.array([0.0, 0.0, 0.0]))
        self.assertEqual(node.node_type, "endpoint")
        
        # Get a non-existent node
        node = self.model.get_node("non_existent")
        
        # Check that None was returned
        self.assertIsNone(node)
        
        # Get all nodes
        nodes = self.model.get_all_nodes()
        
        # Check that all nodes are present
        self.assertEqual(len(nodes), 4)
        self.assertIn("node1", nodes)
        self.assertIn("node2", nodes)
        self.assertIn("node3", nodes)
        self.assertIn("node4", nodes)

    def test_add_get_segment(self):
        """Test adding and getting segments."""
        # Get a segment
        segment = self.model.get_segment(1)
        
        # Check that the segment is correct
        self.assertEqual(segment.segment_id, 1)
        np.testing.assert_array_equal(segment.points, np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ]))
        self.assertEqual(segment.start_node_id, "node1")
        self.assertEqual(segment.end_node_id, "node2")
        
        # Get a non-existent segment
        segment = self.model.get_segment(999)
        
        # Check that None was returned
        self.assertIsNone(segment)
        
        # Get all segments
        segments = self.model.get_all_segments()
        
        # Check that all segments are present
        self.assertEqual(len(segments), 3)
        self.assertIn(1, segments)
        self.assertIn(2, segments)
        self.assertIn(3, segments)

    def test_get_branch_points(self):
        """Test getting branch points."""
        # Get the branch points
        branch_points = self.model.get_branch_points()
        
        # Check that the branch points are correct
        self.assertEqual(len(branch_points), 1)
        np.testing.assert_array_equal(branch_points[0], np.array([3.0, 0.0, 0.0]))

    def test_get_endpoints(self):
        """Test getting endpoints."""
        # Get the endpoints
        endpoints = self.model.get_endpoints()
        
        # Check that the endpoints are correct
        self.assertEqual(len(endpoints), 3)
        
        # Check that all expected endpoints are in the result
        expected_endpoints = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 3.0, 0.0]),
            np.array([3.0, -3.0, 0.0])
        ]
        
        for endpoint in expected_endpoints:
            self.assertTrue(any(np.array_equal(endpoint, ep) for ep in endpoints))

    def test_calculate_total_length(self):
        """Test calculating the total length of all segments."""
        # Calculate the total length
        total_length = self.model.calculate_total_length()
        
        # Check that the total length is correct
        self.assertEqual(total_length, 9.0)  # 3.0 + 3.0 + 3.0

    def test_set_get_property(self):
        """Test setting and getting properties."""
        # Set a property
        self.model.set_property("test_key", "test_value")
        
        # Get the property
        value = self.model.get_property("test_key")
        
        # Check that the property was set correctly
        self.assertEqual(value, "test_value")
        
        # Get a non-existent property with a default value
        value = self.model.get_property("non_existent", "default")
        
        # Check that the default value was returned
        self.assertEqual(value, "default")

    def test_to_dict(self):
        """Test converting a model to a dictionary."""
        # Set some properties
        self.model.set_property("test_key", "test_value")
        self.model.set_property("test_array", np.array([1, 2, 3]))
        
        # Convert to a dictionary
        model_dict = self.model.to_dict()
        
        # Check that the dictionary contains the expected keys
        self.assertIn("model_id", model_dict)
        self.assertIn("name", model_dict)
        self.assertIn("nodes", model_dict)
        self.assertIn("segments", model_dict)
        self.assertIn("properties", model_dict)
        
        # Check that the values are correct
        self.assertEqual(model_dict["model_id"], "model1")
        self.assertEqual(model_dict["name"], "Test Model")
        self.assertEqual(len(model_dict["nodes"]), 4)
        self.assertEqual(len(model_dict["segments"]), 3)
        self.assertEqual(model_dict["properties"]["test_key"], "test_value")
        np.testing.assert_array_equal(model_dict["properties"]["test_array"], np.array([1, 2, 3]))

    def test_from_dict(self):
        """Test creating a model from a dictionary."""
        # Create a dictionary
        model_dict = {
            "model_id": "model2",
            "name": "Test Model 2",
            "nodes": [
                {
                    "node_id": "node5",
                    "position": [0.0, 0.0, 0.0],
                    "node_type": "endpoint",
                    "properties": {}
                },
                {
                    "node_id": "node6",
                    "position": [1.0, 0.0, 0.0],
                    "node_type": "endpoint",
                    "properties": {}
                }
            ],
            "segments": [
                {
                    "segment_id": 4,
                    "points": [
                        [0.0, 0.0, 0.0],
                        [0.5, 0.0, 0.0],
                        [1.0, 0.0, 0.0]
                    ],
                    "start_node_id": "node5",
                    "end_node_id": "node6",
                    "properties": {}
                }
            ],
            "properties": {
                "test_key": "test_value",
                "test_array": [1, 2, 3]
            }
        }
        
        # Create a model from the dictionary
        model = VesselModel.from_dict(model_dict)
        
        # Check that the model was created correctly
        self.assertEqual(model.model_id, "model2")
        self.assertEqual(model.name, "Test Model 2")
        self.assertEqual(len(model.get_all_nodes()), 2)
        self.assertEqual(len(model.get_all_segments()), 1)
        self.assertEqual(model.get_property("test_key"), "test_value")
        np.testing.assert_array_equal(model.get_property("test_array"), np.array([1, 2, 3]))

    def test_save_load_json(self):
        """Test saving and loading a model to/from JSON."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Set some properties
            self.model.set_property("test_key", "test_value")
            self.model.set_property("test_array", np.array([1, 2, 3]))
            
            # Save the model to JSON
            self.model.save_to_json(temp_filename)
            
            # Load the model from JSON
            loaded_model = VesselModel.load_from_json(temp_filename)
            
            # Check that the loaded model is correct
            self.assertEqual(loaded_model.model_id, "model1")
            self.assertEqual(loaded_model.name, "Test Model")
            self.assertEqual(len(loaded_model.get_all_nodes()), 4)
            self.assertEqual(len(loaded_model.get_all_segments()), 3)
            self.assertEqual(loaded_model.get_property("test_key"), "test_value")
            np.testing.assert_array_equal(loaded_model.get_property("test_array"), np.array([1, 2, 3]))
        
        finally:
            # Delete the temporary file
            os.unlink(temp_filename)


if __name__ == '__main__':
    unittest.main()
