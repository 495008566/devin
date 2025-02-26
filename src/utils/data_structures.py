"""
Data structures module.

This module provides data structures for storing vessel geometry and topology information.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import json


class VesselSegment:
    """Class for representing a vessel segment."""

    def __init__(self, segment_id: int, points: np.ndarray, start_node_id: str = None, end_node_id: str = None):
        """
        Initialize a vessel segment.

        Args:
            segment_id: The ID of the segment.
            points: The points defining the segment.
            start_node_id: The ID of the start node.
            end_node_id: The ID of the end node.
        """
        self.segment_id = segment_id
        self.points = points
        self.start_node_id = start_node_id
        self.end_node_id = end_node_id
        self.properties = {}

    def set_property(self, key: str, value: Any):
        """
        Set a property of the segment.

        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property of the segment.

        Args:
            key: The property key.
            default: The default value to return if the property does not exist.

        Returns:
            Any: The property value.
        """
        return self.properties.get(key, default)

    def calculate_length(self) -> float:
        """
        Calculate the length of the segment.

        Returns:
            float: The length of the segment.
        """
        length = 0.0
        for i in range(len(self.points) - 1):
            length += np.linalg.norm(self.points[i+1] - self.points[i])
        return length
        
    def get_point_at_distance(self, distance: float) -> np.ndarray:
        """
        Get a point at a specific distance along the segment.
        
        Args:
            distance: The distance along the segment.
            
        Returns:
            np.ndarray: The point at the specified distance.
        """
        if distance <= 0:
            return self.points[0]
            
        current_distance = 0.0
        for i in range(len(self.points) - 1):
            segment_length = np.linalg.norm(self.points[i+1] - self.points[i])
            if current_distance + segment_length >= distance:
                # Interpolate between the two points
                t = (distance - current_distance) / segment_length
                return self.points[i] + t * (self.points[i+1] - self.points[i])
            current_distance += segment_length
            
        # If the distance is greater than the total length, return the last point
        return self.points[-1]
        
    def get_direction_at_distance(self, distance: float) -> np.ndarray:
        """
        Get the direction at a specific distance along the segment.
        
        Args:
            distance: The distance along the segment.
            
        Returns:
            np.ndarray: The direction at the specified distance.
        """
        if distance <= 0:
            # Return the direction at the start of the segment
            if len(self.points) > 1:
                direction = self.points[1] - self.points[0]
                return direction / np.linalg.norm(direction)
            return np.array([1.0, 0.0, 0.0])  # Default direction if only one point
            
        current_distance = 0.0
        for i in range(len(self.points) - 1):
            segment_length = np.linalg.norm(self.points[i+1] - self.points[i])
            if current_distance + segment_length >= distance:
                # Return the direction of this segment
                direction = self.points[i+1] - self.points[i]
                return direction / np.linalg.norm(direction)
            current_distance += segment_length
            
        # If the distance is greater than the total length, return the direction at the end
        if len(self.points) > 1:
            direction = self.points[-1] - self.points[-2]
            return direction / np.linalg.norm(direction)
        return np.array([1.0, 0.0, 0.0])  # Default direction if only one point

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the segment to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the segment.
        """
        # Convert properties to JSON-serializable format
        properties = {}
        for key, value in self.properties.items():
            if isinstance(value, np.ndarray):
                properties[key] = value.tolist()
            else:
                properties[key] = value
                
        return {
            "segment_id": self.segment_id,
            "points": self.points.tolist() if isinstance(self.points, np.ndarray) else self.points,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "properties": properties
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VesselSegment':
        """
        Create a segment from a dictionary.

        Args:
            data: Dictionary representation of the segment.

        Returns:
            VesselSegment: The created segment.
        """
        points = np.array(data["points"]) if "points" in data else np.array([])
        segment = cls(
            segment_id=data.get("segment_id", -1),
            points=points,
            start_node_id=data.get("start_node_id"),
            end_node_id=data.get("end_node_id")
        )
        
        if "properties" in data:
            for key, value in data["properties"].items():
                segment.set_property(key, value)
        
        return segment


class VesselNode:
    """Class for representing a vessel node (branch point or endpoint)."""

    def __init__(self, node_id: str, position: np.ndarray, node_type: str = "unknown"):
        """
        Initialize a vessel node.

        Args:
            node_id: The ID of the node.
            position: The position of the node.
            node_type: The type of the node (e.g., "branch", "endpoint").
        """
        self.node_id = node_id
        self.position = position
        self.node_type = node_type
        self.properties = {}
        self.connected_segments = []

    def set_property(self, key: str, value: Any):
        """
        Set a property of the node.

        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property of the node.

        Args:
            key: The property key.
            default: The default value to return if the property does not exist.

        Returns:
            Any: The property value.
        """
        return self.properties.get(key, default)

    def add_connected_segment(self, segment_id: int):
        """
        Add a connected segment to the node.

        Args:
            segment_id: The ID of the connected segment.
        """
        if segment_id not in self.connected_segments:
            self.connected_segments.append(segment_id)

    def get_connected_segments(self) -> List[int]:
        """
        Get the connected segments of the node.

        Returns:
            List[int]: The IDs of the connected segments.
        """
        return self.connected_segments

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the node to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the node.
        """
        # Convert properties to JSON-serializable format
        properties = {}
        for key, value in self.properties.items():
            if isinstance(value, np.ndarray):
                properties[key] = value.tolist()
            else:
                properties[key] = value
                
        return {
            "node_id": self.node_id,
            "position": self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            "node_type": self.node_type,
            "properties": properties,
            "connected_segments": self.connected_segments
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VesselNode':
        """
        Create a node from a dictionary.

        Args:
            data: Dictionary representation of the node.

        Returns:
            VesselNode: The created node.
        """
        position = np.array(data["position"]) if "position" in data else np.array([0, 0, 0])
        node = cls(
            node_id=data.get("node_id", ""),
            position=position,
            node_type=data.get("node_type", "unknown")
        )
        
        if "properties" in data:
            for key, value in data["properties"].items():
                node.set_property(key, value)
        
        if "connected_segments" in data:
            for segment_id in data["connected_segments"]:
                node.add_connected_segment(segment_id)
        
        return node


class VesselModel:
    """Class for representing a complete vessel model."""

    def __init__(self, model_id: str = "", name: str = ""):
        """
        Initialize a vessel model.

        Args:
            model_id: The ID of the model.
            name: The name of the model.
        """
        self.model_id = model_id
        self.name = name
        self.segments = {}  # segment_id -> VesselSegment
        self.nodes = {}  # node_id -> VesselNode
        self.properties = {}

    def set_property(self, key: str, value: Any):
        """
        Set a property of the model.

        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property of the model.

        Args:
            key: The property key.
            default: The default value to return if the property does not exist.

        Returns:
            Any: The property value.
        """
        return self.properties.get(key, default)

    def add_segment(self, segment: VesselSegment):
        """
        Add a segment to the model.

        Args:
            segment: The segment to add.
        """
        self.segments[segment.segment_id] = segment
        
        # Update the connected segments of the nodes
        if segment.start_node_id and segment.start_node_id in self.nodes:
            self.nodes[segment.start_node_id].add_connected_segment(segment.segment_id)
        
        if segment.end_node_id and segment.end_node_id in self.nodes:
            self.nodes[segment.end_node_id].add_connected_segment(segment.segment_id)

    def add_node(self, node: VesselNode):
        """
        Add a node to the model.

        Args:
            node: The node to add.
        """
        self.nodes[node.node_id] = node

    def get_segment(self, segment_id: int) -> Optional[VesselSegment]:
        """
        Get a segment by ID.

        Args:
            segment_id: The ID of the segment.

        Returns:
            Optional[VesselSegment]: The segment, or None if not found.
        """
        return self.segments.get(segment_id)

    def get_node(self, node_id: str) -> Optional[VesselNode]:
        """
        Get a node by ID.

        Args:
            node_id: The ID of the node.

        Returns:
            Optional[VesselNode]: The node, or None if not found.
        """
        return self.nodes.get(node_id)

    def get_all_segments(self) -> Dict[int, VesselSegment]:
        """
        Get all segments.

        Returns:
            Dict[int, VesselSegment]: Dictionary mapping segment IDs to segments.
        """
        return self.segments

    def get_all_nodes(self) -> Dict[str, VesselNode]:
        """
        Get all nodes.

        Returns:
            Dict[str, VesselNode]: Dictionary mapping node IDs to nodes.
        """
        return self.nodes

    def get_branch_points(self) -> List[np.ndarray]:
        """
        Get all branch points.

        Returns:
            List[np.ndarray]: List of branch point positions.
        """
        return [node.position for node_id, node in self.nodes.items() if node.node_type == "branch"]

    def get_endpoints(self) -> List[np.ndarray]:
        """
        Get all endpoints.

        Returns:
            List[np.ndarray]: List of endpoint positions.
        """
        endpoints = [node.position for node_id, node in self.nodes.items() if node.node_type == "endpoint"]
        
        # If no endpoints were found but we have centerline data, use the first and last points
        if not endpoints:
            centerline = self.get_property("centerline")
            if centerline is not None and len(centerline) >= 2:
                return [centerline[0], centerline[-1]]
        
        return endpoints

    def calculate_total_length(self) -> float:
        """
        Calculate the total length of all segments.

        Returns:
            float: The total length.
        """
        return sum(segment.calculate_length() for segment in self.segments.values())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the model to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        # Convert properties to JSON-serializable format
        properties = {}
        for key, value in self.properties.items():
            if isinstance(value, np.ndarray):
                properties[key] = value.tolist()
            elif isinstance(value, list):
                # Handle lists that might contain numpy arrays
                properties[key] = self._convert_list_for_json(value)
            elif isinstance(value, dict):
                # Handle dictionaries that might contain numpy arrays
                properties[key] = self._convert_dict_for_json(value)
            else:
                properties[key] = value
                
        return {
            "model_id": self.model_id,
            "name": self.name,
            "properties": properties,
            "segments": [segment.to_dict() for segment in self.segments.values()],
            "nodes": [node.to_dict() for node in self.nodes.values()]
        }
        
    def _convert_list_for_json(self, lst):
        """
        Convert a list to a JSON-serializable format.
        
        Args:
            lst: The list to convert.
            
        Returns:
            A JSON-serializable list.
        """
        result = []
        for item in lst:
            if isinstance(item, np.ndarray):
                result.append(item.tolist())
            elif isinstance(item, list):
                result.append(self._convert_list_for_json(item))
            elif isinstance(item, dict):
                result.append(self._convert_dict_for_json(item))
            elif isinstance(item, np.integer):
                result.append(int(item))
            elif isinstance(item, np.floating):
                result.append(float(item))
            else:
                result.append(item)
        return result
        
    def _convert_dict_for_json(self, d):
        """
        Convert a dictionary to a JSON-serializable format.
        
        Args:
            d: The dictionary to convert.
            
        Returns:
            A JSON-serializable dictionary.
        """
        result = {}
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, list):
                result[key] = self._convert_list_for_json(value)
            elif isinstance(value, dict):
                result[key] = self._convert_dict_for_json(value)
            elif isinstance(value, np.integer):
                result[key] = int(value)
            elif isinstance(value, np.floating):
                result[key] = float(value)
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VesselModel':
        """
        Create a model from a dictionary.

        Args:
            data: Dictionary representation of the model.

        Returns:
            VesselModel: The created model.
        """
        model = cls(
            model_id=data.get("model_id", ""),
            name=data.get("name", "")
        )
        
        if "properties" in data:
            for key, value in data["properties"].items():
                model.set_property(key, value)
        
        if "nodes" in data:
            # Handle both list and dictionary formats
            if isinstance(data["nodes"], list):
                for node_data in data["nodes"]:
                    node = VesselNode.from_dict(node_data)
                    model.add_node(node)
            else:
                for node_id, node_data in data["nodes"].items():
                    node = VesselNode.from_dict(node_data)
                    model.add_node(node)
        
        if "segments" in data:
            # Handle both list and dictionary formats
            if isinstance(data["segments"], list):
                for segment_data in data["segments"]:
                    segment = VesselSegment.from_dict(segment_data)
                    model.add_segment(segment)
            else:
                for segment_id, segment_data in data["segments"].items():
                    segment = VesselSegment.from_dict(segment_data)
                    model.add_segment(segment)
        
        return model

    def save_to_json(self, filename: str):
        """
        Save the model to a JSON file.

        Args:
            filename: The filename to save to.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filename: str) -> 'VesselModel':
        """
        Load a model from a JSON file.

        Args:
            filename: The filename to load from.

        Returns:
            VesselModel: The loaded model.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


class VesselDatabase:
    """Class for managing a collection of vessel models."""

    def __init__(self):
        """Initialize a vessel database."""
        self.models = {}  # model_id -> VesselModel
        self.metadata = {}

    def add_model(self, model: VesselModel):
        """
        Add a model to the database.

        Args:
            model: The model to add.
        """
        self.models[model.model_id] = model

    def get_model(self, model_id: str) -> Optional[VesselModel]:
        """
        Get a model by ID.

        Args:
            model_id: The ID of the model.

        Returns:
            Optional[VesselModel]: The model, or None if not found.
        """
        return self.models.get(model_id)

    def get_all_models(self) -> Dict[str, VesselModel]:
        """
        Get all models.

        Returns:
            Dict[str, VesselModel]: Dictionary mapping model IDs to models.
        """
        return self.models

    def set_metadata(self, key: str, value: Any):
        """
        Set a metadata value.

        Args:
            key: The metadata key.
            value: The metadata value.
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get a metadata value.

        Args:
            key: The metadata key.
            default: The default value to return if the metadata does not exist.

        Returns:
            Any: The metadata value.
        """
        return self.metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the database to a dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the database.
        """
        return {
            "metadata": self.metadata,
            "models": {model_id: model.to_dict() for model_id, model in self.models.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VesselDatabase':
        """
        Create a database from a dictionary.

        Args:
            data: Dictionary representation of the database.

        Returns:
            VesselDatabase: The created database.
        """
        database = cls()
        
        if "metadata" in data:
            for key, value in data["metadata"].items():
                database.set_metadata(key, value)
        
        if "models" in data:
            for model_id, model_data in data["models"].items():
                model = VesselModel.from_dict(model_data)
                database.add_model(model)
        
        return database

    def save_to_json(self, filename: str):
        """
        Save the database to a JSON file.

        Args:
            filename: The filename to save to.
        """
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filename: str) -> 'VesselDatabase':
        """
        Load a database from a JSON file.

        Args:
            filename: The filename to load from.

        Returns:
            VesselDatabase: The loaded database.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
