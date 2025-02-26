"""
Data Structures module for the blood vessel analysis system.

This module provides data structures for representing vessel geometry and topology.
"""

import os
import json
import uuid
import numpy as np

class VesselNode:
    """Class for representing a vessel node (branch point or endpoint)."""
    
    def __init__(self, node_id, position, node_type="unknown"):
        """
        Initialize a vessel node.
        
        Args:
            node_id: The ID of the node.
            position: The position of the node.
            node_type: The type of the node (e.g., "branch", "endpoint").
        """
        self.node_id = node_id
        self.position = np.array(position)
        self.node_type = node_type
        self.properties = {}
        self.connected_segments = []
    
    def set_property(self, key, value):
        """
        Set a property of the node.
        
        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value
    
    def get_property(self, key, default=None):
        """
        Get a property of the node.
        
        Args:
            key: The property key.
            default: The default value to return if the property does not exist.
            
        Returns:
            The property value.
        """
        return self.properties.get(key, default)
    
    def add_connected_segment(self, segment_id):
        """
        Add a connected segment to the node.
        
        Args:
            segment_id: The ID of the connected segment.
        """
        if segment_id not in self.connected_segments:
            self.connected_segments.append(segment_id)
    
    def get_connected_segments(self):
        """
        Get the connected segments of the node.
        
        Returns:
            The IDs of the connected segments.
        """
        return self.connected_segments
    
    def to_dict(self):
        """
        Convert the node to a dictionary.
        
        Returns:
            Dictionary representation of the node.
        """
        return {
            "node_id": self.node_id,
            "position": self.position.tolist(),
            "node_type": self.node_type,
            "properties": self.properties,
            "connected_segments": self.connected_segments
        }

class VesselSegment:
    """Class for representing a vessel segment."""
    
    def __init__(self, segment_id, points, start_node_id=None, end_node_id=None):
        """
        Initialize a vessel segment.
        
        Args:
            segment_id: The ID of the segment.
            points: The points defining the segment.
            start_node_id: The ID of the start node.
            end_node_id: The ID of the end node.
        """
        self.segment_id = segment_id
        self.points = np.array(points)
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.properties = {}
    
    def set_property(self, key, value):
        """
        Set a property of the segment.
        
        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value
    
    def get_property(self, key, default=None):
        """
        Get a property of the segment.
        
        Args:
            key: The property key.
            default: The default value to return if the property does not exist.
            
        Returns:
            The property value.
        """
        return self.properties.get(key, default)
    
    def calculate_length(self):
        """
        Calculate the length of the segment.
        
        Returns:
            The length of the segment.
        """
        length = 0.0
        for i in range(len(self.points) - 1):
            length += np.linalg.norm(self.points[i+1] - self.points[i])
        return length
    
    def get_point_at_distance(self, distance):
        """
        Get a point at a specific distance along the segment.
        
        Args:
            distance: The distance along the segment.
            
        Returns:
            The point at the specified distance.
        """
        if distance <= 0:
            return self.points[0]
        
        if distance >= self.calculate_length():
            return self.points[-1]
        
        current_distance = 0.0
        for i in range(len(self.points) - 1):
            segment_length = np.linalg.norm(self.points[i+1] - self.points[i])
            if current_distance + segment_length >= distance:
                # Interpolate between the two points
                t = (distance - current_distance) / segment_length
                return self.points[i] + t * (self.points[i+1] - self.points[i])
            current_distance += segment_length
        
        return self.points[-1]
    
    def get_direction_at_distance(self, distance):
        """
        Get the direction at a specific distance along the segment.
        
        Args:
            distance: The distance along the segment.
            
        Returns:
            The direction at the specified distance.
        """
        if distance <= 0:
            if len(self.points) > 1:
                direction = self.points[1] - self.points[0]
                return direction / np.linalg.norm(direction)
            return np.array([1, 0, 0])
        
        if distance >= self.calculate_length():
            if len(self.points) > 1:
                direction = self.points[-1] - self.points[-2]
                return direction / np.linalg.norm(direction)
            return np.array([1, 0, 0])
        
        current_distance = 0.0
        for i in range(len(self.points) - 1):
            segment_length = np.linalg.norm(self.points[i+1] - self.points[i])
            if current_distance + segment_length >= distance:
                direction = self.points[i+1] - self.points[i]
                return direction / np.linalg.norm(direction)
            current_distance += segment_length
        
        if len(self.points) > 1:
            direction = self.points[-1] - self.points[-2]
            return direction / np.linalg.norm(direction)
        return np.array([1, 0, 0])
    
    def to_dict(self):
        """
        Convert the segment to a dictionary.
        
        Returns:
            Dictionary representation of the segment.
        """
        return {
            "segment_id": self.segment_id,
            "points": self.points.tolist(),
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "properties": self.properties
        }

class VesselModel:
    """Class for representing a vessel model."""
    
    def __init__(self, model_id=None):
        """
        Initialize a vessel model.
        
        Args:
            model_id: The ID of the model.
        """
        self.model_id = model_id if model_id is not None else str(uuid.uuid4())
        self.nodes = {}
        self.segments = {}
        self.properties = {}
    
    def add_node(self, node):
        """
        Add a node to the model.
        
        Args:
            node: The node to add.
        """
        self.nodes[node.node_id] = node
    
    def add_segment(self, segment):
        """
        Add a segment to the model.
        
        Args:
            segment: The segment to add.
        """
        self.segments[segment.segment_id] = segment
    
    def get_node(self, node_id):
        """
        Get a node from the model.
        
        Args:
            node_id: The ID of the node.
            
        Returns:
            The node with the specified ID, or None if not found.
        """
        return self.nodes.get(node_id)
    
    def get_segment(self, segment_id):
        """
        Get a segment from the model.
        
        Args:
            segment_id: The ID of the segment.
            
        Returns:
            The segment with the specified ID, or None if not found.
        """
        return self.segments.get(segment_id)
    
    def get_all_nodes(self):
        """
        Get all nodes in the model.
        
        Returns:
            Dictionary of nodes.
        """
        return self.nodes
    
    def get_all_segments(self):
        """
        Get all segments in the model.
        
        Returns:
            Dictionary of segments.
        """
        return self.segments
    
    def get_branch_points(self):
        """
        Get all branch points in the model.
        
        Returns:
            List of branch point positions.
        """
        branch_points = []
        for node_id, node in self.nodes.items():
            if node.node_type == "branch":
                branch_points.append(node.position)
        return branch_points
    
    def get_endpoints(self):
        """
        Get all endpoints in the model.
        
        Returns:
            List of endpoint positions.
        """
        endpoints = []
        for node_id, node in self.nodes.items():
            if node.node_type == "endpoint":
                endpoints.append(node.position)
        return endpoints
    
    def set_property(self, key, value):
        """
        Set a property of the model.
        
        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value
    
    def get_property(self, key, default=None):
        """
        Get a property of the model.
        
        Args:
            key: The property key.
            default: The default value to return if the property does not exist.
            
        Returns:
            The property value.
        """
        return self.properties.get(key, default)
    
    def to_dict(self):
        """
        Convert the model to a dictionary.
        
        Returns:
            Dictionary representation of the model.
        """
        return {
            "model_id": self.model_id,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "segments": {segment_id: segment.to_dict() for segment_id, segment in self.segments.items()},
            "properties": self.properties
        }
    
    def to_json(self, indent=2):
        """
        Convert the model to a JSON string.
        
        Args:
            indent: The indentation level for the JSON string.
            
        Returns:
            JSON string representation of the model.
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, filename):
        """
        Save the model to a JSON file.
        
        Args:
            filename: The name of the file to save to.
            
        Returns:
            bool: True if the file was saved successfully, False otherwise.
        """
        try:
            with open(filename, 'w') as f:
                f.write(self.to_json())
            return True
        except Exception as e:
            print(f"Error saving model to file: {e}")
            return False
    
    @classmethod
    def from_dict(cls, data):
        """
        Create a model from a dictionary.
        
        Args:
            data: Dictionary representation of the model.
            
        Returns:
            VesselModel: The created model.
        """
        model = cls(model_id=data.get("model_id"))
        
        # Add nodes
        for node_id, node_data in data.get("nodes", {}).items():
            node = VesselNode(
                node_id=node_data.get("node_id"),
                position=node_data.get("position"),
                node_type=node_data.get("node_type")
            )
            for key, value in node_data.get("properties", {}).items():
                node.set_property(key, value)
            for segment_id in node_data.get("connected_segments", []):
                node.add_connected_segment(segment_id)
            model.add_node(node)
        
        # Add segments
        for segment_id, segment_data in data.get("segments", {}).items():
            segment = VesselSegment(
                segment_id=segment_data.get("segment_id"),
                points=segment_data.get("points"),
                start_node_id=segment_data.get("start_node_id"),
                end_node_id=segment_data.get("end_node_id")
            )
            for key, value in segment_data.get("properties", {}).items():
                segment.set_property(key, value)
            model.add_segment(segment)
        
        # Set properties
        for key, value in data.get("properties", {}).items():
            model.set_property(key, value)
        
        return model
    
    @classmethod
    def from_json(cls, json_str):
        """
        Create a model from a JSON string.
        
        Args:
            json_str: JSON string representation of the model.
            
        Returns:
            VesselModel: The created model.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except Exception as e:
            print(f"Error creating model from JSON: {e}")
            return None
    
    @classmethod
    def from_file(cls, filename):
        """
        Create a model from a JSON file.
        
        Args:
            filename: The name of the file to load from.
            
        Returns:
            VesselModel: The created model.
        """
        try:
            with open(filename, 'r') as f:
                json_str = f.read()
            return cls.from_json(json_str)
        except Exception as e:
            print(f"Error loading model from file: {e}")
            return None
