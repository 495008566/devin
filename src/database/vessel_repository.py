"""
Vessel repository module.

This module provides functionality for storing and retrieving blood vessel data in MySQL.
"""

import numpy as np
import json
from typing import Dict, Any, List, Tuple, Optional, Union
import os

from src.database.mysql_connector import MySQLConnector
from src.utils.data_structures import VesselModel, VesselNode, VesselSegment


class VesselRepository:
    """Class for storing and retrieving blood vessel data in MySQL."""

    def __init__(self, connector: MySQLConnector = None):
        """
        Initialize the vessel repository.

        Args:
            connector: The MySQL connector to use.
        """
        self.connector = connector or MySQLConnector()

    def connect(self) -> bool:
        """
        Connect to the database.

        Returns:
            bool: True if the connection was successful, False otherwise.
        """
        if not self.connector.is_connected():
            return self.connector.connect()
        return True

    def disconnect(self):
        """Disconnect from the database."""
        self.connector.disconnect()

    def create_tables(self) -> bool:
        """
        Create the database tables.

        Returns:
            bool: True if the tables were created successfully, False otherwise.
        """
        return self.connector.create_tables()

    def save_model(self, model: VesselModel) -> bool:
        """
        Save a vessel model to the database.

        Args:
            model: The vessel model to save.

        Returns:
            bool: True if the model was saved successfully, False otherwise.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return False

        try:
            # Extract model properties
            model_id = model.model_id
            name = model.name
            filename = model.get_property("filename", "")
            surface_area = model.get_property("surface_area", 0.0)
            volume = model.get_property("mesh_volume", 0.0)
            centerline_length = model.calculate_total_length()
            num_branch_points = len(model.get_branch_points())
            num_endpoints = len(model.get_endpoints())
            num_segments = len(model.get_all_segments())
            
            # Convert properties to JSON
            properties = {k: v for k, v in model.properties.items() 
                         if k not in ["filename", "surface_area", "mesh_volume", 
                                     "centerline", "branch_points", "endpoints", "segments",
                                     "segment_lengths", "segment_diameters", "cross_sections"]}
            
            # Convert numpy arrays to lists
            for k, v in properties.items():
                if isinstance(v, np.ndarray):
                    properties[k] = v.tolist()
            
            properties_json = json.dumps(properties)
            
            # Insert the model
            query = """
                INSERT INTO models 
                (model_id, name, filename, surface_area, volume, centerline_length, 
                num_branch_points, num_endpoints, num_segments, properties)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                name = VALUES(name),
                filename = VALUES(filename),
                surface_area = VALUES(surface_area),
                volume = VALUES(volume),
                centerline_length = VALUES(centerline_length),
                num_branch_points = VALUES(num_branch_points),
                num_endpoints = VALUES(num_endpoints),
                num_segments = VALUES(num_segments),
                properties = VALUES(properties)
            """
            
            params = (model_id, name, filename, surface_area, volume, centerline_length,
                     num_branch_points, num_endpoints, num_segments, properties_json)
            
            self.connector.execute_query(query, params)
            
            # Save nodes
            for node_id, node in model.get_all_nodes().items():
                self._save_node(node, model_id)
            
            # Save segments
            for segment_id, segment in model.get_all_segments().items():
                self._save_segment(segment, model_id)
            
            # Save cross-sections
            cross_sections = model.get_property("cross_sections", [])
            if cross_sections:
                self._save_cross_sections(cross_sections, model_id)
            
            # Save bifurcation angles
            bifurcation_angles = model.get_property("bifurcation_angles", {})
            if bifurcation_angles:
                self._save_bifurcation_angles(bifurcation_angles, model_id)
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def _save_node(self, node: VesselNode, model_id: str) -> bool:
        """
        Save a vessel node to the database.

        Args:
            node: The vessel node to save.
            model_id: The ID of the model the node belongs to.

        Returns:
            bool: True if the node was saved successfully, False otherwise.
        """
        try:
            # Extract node properties
            node_id = node.node_id
            node_type = node.node_type
            position = node.position
            
            # Convert properties to JSON
            properties = {k: v for k, v in node.properties.items()}
            
            # Convert numpy arrays to lists
            for k, v in properties.items():
                if isinstance(v, np.ndarray):
                    properties[k] = v.tolist()
            
            properties_json = json.dumps(properties)
            
            # Insert the node
            query = """
                INSERT INTO nodes 
                (node_id, model_id, node_type, position_x, position_y, position_z, properties)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                node_type = VALUES(node_type),
                position_x = VALUES(position_x),
                position_y = VALUES(position_y),
                position_z = VALUES(position_z),
                properties = VALUES(properties)
            """
            
            params = (node_id, model_id, node_type, position[0], position[1], position[2], properties_json)
            
            return self.connector.execute_query(query, params)
        
        except Exception as e:
            print(f"Error saving node: {e}")
            return False

    def _save_segment(self, segment: VesselSegment, model_id: str) -> bool:
        """
        Save a vessel segment to the database.

        Args:
            segment: The vessel segment to save.
            model_id: The ID of the model the segment belongs to.

        Returns:
            bool: True if the segment was saved successfully, False otherwise.
        """
        try:
            # Extract segment properties
            segment_id = segment.segment_id
            start_node_id = segment.start_node_id
            end_node_id = segment.end_node_id
            length = segment.get_property("length", segment.calculate_length())
            diameter = segment.get_property("diameter", 0.0)
            tortuosity = segment.get_property("tortuosity", 0.0)
            
            # Convert properties to JSON
            properties = {k: v for k, v in segment.properties.items() 
                         if k not in ["length", "diameter", "tortuosity"]}
            
            # Convert numpy arrays to lists
            for k, v in properties.items():
                if isinstance(v, np.ndarray):
                    properties[k] = v.tolist()
            
            properties_json = json.dumps(properties)
            
            # Insert the segment
            query = """
                INSERT INTO segments 
                (segment_id, model_id, start_node_id, end_node_id, length, diameter, tortuosity, properties)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                start_node_id = VALUES(start_node_id),
                end_node_id = VALUES(end_node_id),
                length = VALUES(length),
                diameter = VALUES(diameter),
                tortuosity = VALUES(tortuosity),
                properties = VALUES(properties)
            """
            
            params = (segment_id, model_id, start_node_id, end_node_id, length, diameter, tortuosity, properties_json)
            
            success = self.connector.execute_query(query, params)
            
            if success:
                # Save segment points
                self._save_segment_points(segment.points, segment_id, model_id)
            
            return success
        
        except Exception as e:
            print(f"Error saving segment: {e}")
            return False

    def _save_segment_points(self, points: np.ndarray, segment_id: int, model_id: str) -> bool:
        """
        Save segment points to the database.

        Args:
            points: The segment points to save.
            segment_id: The ID of the segment the points belong to.
            model_id: The ID of the model the segment belongs to.

        Returns:
            bool: True if the points were saved successfully, False otherwise.
        """
        try:
            # Delete existing points
            query = """
                DELETE FROM segment_points
                WHERE segment_id = %s AND model_id = %s
            """
            
            self.connector.execute_query(query, (segment_id, model_id))
            
            # Insert the points
            for i, point in enumerate(points):
                query = """
                    INSERT INTO segment_points 
                    (segment_id, model_id, point_index, position_x, position_y, position_z)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                
                params = (segment_id, model_id, i, point[0], point[1], point[2])
                
                self.connector.execute_query(query, params)
            
            return True
        
        except Exception as e:
            print(f"Error saving segment points: {e}")
            return False

    def _save_cross_sections(self, cross_sections: List[Dict[str, Any]], model_id: str) -> bool:
        """
        Save cross-sections to the database.

        Args:
            cross_sections: The cross-sections to save.
            model_id: The ID of the model the cross-sections belong to.

        Returns:
            bool: True if the cross-sections were saved successfully, False otherwise.
        """
        try:
            # Delete existing cross-sections
            query = """
                DELETE FROM cross_sections
                WHERE model_id = %s
            """
            
            self.connector.execute_query(query, (model_id,))
            
            # Insert the cross-sections
            for cs in cross_sections:
                if "error" in cs:
                    continue
                
                position = cs.get("position", [0, 0, 0])
                segment_id = cs.get("segment_id", 0)
                area = cs.get("area", 0.0)
                perimeter = cs.get("perimeter", 0.0)
                equivalent_diameter = cs.get("equivalent_diameter", 0.0)
                circularity = cs.get("circularity", 0.0)
                
                # Convert properties to JSON
                properties = {k: v for k, v in cs.items() 
                             if k not in ["position", "segment_id", "area", "perimeter", 
                                         "equivalent_diameter", "circularity", "error"]}
                
                # Convert numpy arrays to lists
                for k, v in properties.items():
                    if isinstance(v, np.ndarray):
                        properties[k] = v.tolist()
                
                properties_json = json.dumps(properties)
                
                query = """
                    INSERT INTO cross_sections 
                    (model_id, segment_id, position_x, position_y, position_z, 
                    area, perimeter, equivalent_diameter, circularity, properties)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                params = (model_id, segment_id, position[0], position[1], position[2],
                         area, perimeter, equivalent_diameter, circularity, properties_json)
                
                self.connector.execute_query(query, params)
            
            return True
        
        except Exception as e:
            print(f"Error saving cross-sections: {e}")
            return False

    def _save_bifurcation_angles(self, bifurcation_angles: Dict[str, List[float]], model_id: str) -> bool:
        """
        Save bifurcation angles to the database.

        Args:
            bifurcation_angles: The bifurcation angles to save.
            model_id: The ID of the model the bifurcation angles belong to.

        Returns:
            bool: True if the bifurcation angles were saved successfully, False otherwise.
        """
        try:
            # Delete existing bifurcation angles
            query = """
                DELETE FROM bifurcation_angles
                WHERE model_id = %s
            """
            
            self.connector.execute_query(query, (model_id,))
            
            # Insert the bifurcation angles
            for node_id, angles in bifurcation_angles.items():
                # Get the segments connected to the node
                query = """
                    SELECT segment_id
                    FROM segments
                    WHERE model_id = %s AND (start_node_id = %s OR end_node_id = %s)
                """
                
                segments = self.connector.fetch_all(query, (model_id, node_id, node_id))
                
                if len(segments) < 2:
                    continue
                
                # Insert the bifurcation angles
                for i in range(len(segments)):
                    for j in range(i+1, len(segments)):
                        if i < len(angles):
                            segment1_id = segments[i][0]
                            segment2_id = segments[j][0]
                            angle = angles[i]
                            
                            query = """
                                INSERT INTO bifurcation_angles 
                                (model_id, node_id, segment1_id, segment2_id, angle)
                                VALUES (%s, %s, %s, %s, %s)
                            """
                            
                            params = (model_id, node_id, segment1_id, segment2_id, angle)
                            
                            self.connector.execute_query(query, params)
            
            return True
        
        except Exception as e:
            print(f"Error saving bifurcation angles: {e}")
            return False

    def get_model(self, model_id: str) -> Optional[VesselModel]:
        """
        Get a vessel model from the database.

        Args:
            model_id: The ID of the model to get.

        Returns:
            Optional[VesselModel]: The vessel model, or None if not found.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return None

        try:
            # Get the model
            query = """
                SELECT model_id, name, filename, surface_area, volume, centerline_length, 
                num_branch_points, num_endpoints, num_segments, properties
                FROM models
                WHERE model_id = %s
            """
            
            model_data = self.connector.fetch_one(query, (model_id,))
            
            if not model_data:
                return None
            
            # Create the model
            model = VesselModel(model_id=model_data[0], name=model_data[1])
            
            # Set model properties
            model.set_property("filename", model_data[2])
            model.set_property("surface_area", model_data[3])
            model.set_property("mesh_volume", model_data[4])
            
            # Parse the properties JSON
            properties = json.loads(model_data[9]) if model_data[9] else {}
            
            for k, v in properties.items():
                model.set_property(k, v)
            
            # Get the nodes
            nodes = self._get_nodes(model_id)
            
            for node in nodes:
                model.add_node(node)
            
            # Get the segments
            segments = self._get_segments(model_id)
            
            for segment in segments:
                model.add_segment(segment)
            
            # Get the cross-sections
            cross_sections = self._get_cross_sections(model_id)
            
            if cross_sections:
                model.set_property("cross_sections", cross_sections)
            
            # Get the bifurcation angles
            bifurcation_angles = self._get_bifurcation_angles(model_id)
            
            if bifurcation_angles:
                model.set_property("bifurcation_angles", bifurcation_angles)
            
            return model
        
        except Exception as e:
            print(f"Error getting model: {e}")
            return None

    def _get_nodes(self, model_id: str) -> List[VesselNode]:
        """
        Get the nodes of a model from the database.

        Args:
            model_id: The ID of the model to get the nodes for.

        Returns:
            List[VesselNode]: The nodes of the model.
        """
        try:
            # Get the nodes
            query = """
                SELECT node_id, node_type, position_x, position_y, position_z, properties
                FROM nodes
                WHERE model_id = %s
            """
            
            nodes_data = self.connector.fetch_all(query, (model_id,))
            
            nodes = []
            
            for node_data in nodes_data:
                # Create the node
                position = np.array([node_data[2], node_data[3], node_data[4]])
                node = VesselNode(node_id=node_data[0], position=position, node_type=node_data[1])
                
                # Parse the properties JSON
                properties = json.loads(node_data[5]) if node_data[5] else {}
                
                for k, v in properties.items():
                    node.set_property(k, v)
                
                nodes.append(node)
            
            return nodes
        
        except Exception as e:
            print(f"Error getting nodes: {e}")
            return []

    def _get_segments(self, model_id: str) -> List[VesselSegment]:
        """
        Get the segments of a model from the database.

        Args:
            model_id: The ID of the model to get the segments for.

        Returns:
            List[VesselSegment]: The segments of the model.
        """
        try:
            # Get the segments
            query = """
                SELECT segment_id, start_node_id, end_node_id, length, diameter, tortuosity, properties
                FROM segments
                WHERE model_id = %s
            """
            
            segments_data = self.connector.fetch_all(query, (model_id,))
            
            segments = []
            
            for segment_data in segments_data:
                # Get the segment points
                points = self._get_segment_points(segment_data[0], model_id)
                
                # Create the segment
                segment = VesselSegment(
                    segment_id=segment_data[0],
                    points=points,
                    start_node_id=segment_data[1],
                    end_node_id=segment_data[2]
                )
                
                # Set segment properties
                segment.set_property("length", segment_data[3])
                segment.set_property("diameter", segment_data[4])
                segment.set_property("tortuosity", segment_data[5])
                
                # Parse the properties JSON
                properties = json.loads(segment_data[6]) if segment_data[6] else {}
                
                for k, v in properties.items():
                    segment.set_property(k, v)
                
                segments.append(segment)
            
            return segments
        
        except Exception as e:
            print(f"Error getting segments: {e}")
            return []

    def _get_segment_points(self, segment_id: int, model_id: str) -> np.ndarray:
        """
        Get the points of a segment from the database.

        Args:
            segment_id: The ID of the segment to get the points for.
            model_id: The ID of the model the segment belongs to.

        Returns:
            np.ndarray: The points of the segment.
        """
        try:
            # Get the segment points
            query = """
                SELECT point_index, position_x, position_y, position_z
                FROM segment_points
                WHERE segment_id = %s AND model_id = %s
                ORDER BY point_index
            """
            
            points_data = self.connector.fetch_all(query, (segment_id, model_id))
            
            # Create the points array
            points = np.zeros((len(points_data), 3))
            
            for i, point_data in enumerate(points_data):
                points[i] = [point_data[1], point_data[2], point_data[3]]
            
            return points
        
        except Exception as e:
            print(f"Error getting segment points: {e}")
            return np.array([])

    def _get_cross_sections(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get the cross-sections of a model from the database.

        Args:
            model_id: The ID of the model to get the cross-sections for.

        Returns:
            List[Dict[str, Any]]: The cross-sections of the model.
        """
        try:
            # Get the cross-sections
            query = """
                SELECT cross_section_id, segment_id, position_x, position_y, position_z, 
                area, perimeter, equivalent_diameter, circularity, properties
                FROM cross_sections
                WHERE model_id = %s
                ORDER BY cross_section_id
            """
            
            cross_sections_data = self.connector.fetch_all(query, (model_id,))
            
            cross_sections = []
            
            for cs_data in cross_sections_data:
                # Create the cross-section
                cross_section = {
                    "cross_section_id": cs_data[0],
                    "segment_id": cs_data[1],
                    "position": [cs_data[2], cs_data[3], cs_data[4]],
                    "area": cs_data[5],
                    "perimeter": cs_data[6],
                    "equivalent_diameter": cs_data[7],
                    "circularity": cs_data[8]
                }
                
                # Parse the properties JSON
                properties = json.loads(cs_data[9]) if cs_data[9] else {}
                
                # Add the properties to the cross-section
                cross_section.update(properties)
                
                cross_sections.append(cross_section)
            
            return cross_sections
        
        except Exception as e:
            print(f"Error getting cross-sections: {e}")
            return []

    def _get_bifurcation_angles(self, model_id: str) -> Dict[str, List[float]]:
        """
        Get the bifurcation angles of a model from the database.

        Args:
            model_id: The ID of the model to get the bifurcation angles for.

        Returns:
            Dict[str, List[float]]: The bifurcation angles of the model.
        """
        try:
            # Get the bifurcation angles
            query = """
                SELECT node_id, angle
                FROM bifurcation_angles
                WHERE model_id = %s
                ORDER BY node_id, bifurcation_id
            """
            
            angles_data = self.connector.fetch_all(query, (model_id,))
            
            bifurcation_angles = {}
            
            for angle_data in angles_data:
                node_id = angle_data[0]
                angle = angle_data[1]
                
                if node_id not in bifurcation_angles:
                    bifurcation_angles[node_id] = []
                
                bifurcation_angles[node_id].append(angle)
            
            return bifurcation_angles
        
        except Exception as e:
            print(f"Error getting bifurcation angles: {e}")
            return {}

    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get all models from the database.

        Returns:
            List[Dict[str, Any]]: The models.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return []

        try:
            # Get the models
            query = """
                SELECT model_id, name, filename, surface_area, volume, centerline_length, 
                num_branch_points, num_endpoints, num_segments, created_at, updated_at
                FROM models
                ORDER BY updated_at DESC
            """
            
            models_data = self.connector.fetch_all(query)
            
            models = []
            
            for model_data in models_data:
                # Create the model
                model = {
                    "model_id": model_data[0],
                    "name": model_data[1],
                    "filename": model_data[2],
                    "surface_area": model_data[3],
                    "volume": model_data[4],
                    "centerline_length": model_data[5],
                    "num_branch_points": model_data[6],
                    "num_endpoints": model_data[7],
                    "num_segments": model_data[8],
                    "created_at": model_data[9],
                    "updated_at": model_data[10]
                }
                
                models.append(model)
            
            return models
        
        except Exception as e:
            print(f"Error getting models: {e}")
            return []

    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from the database.

        Args:
            model_id: The ID of the model to delete.

        Returns:
            bool: True if the model was deleted successfully, False otherwise.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return False

        try:
            # Delete the model
            query = """
                DELETE FROM models
                WHERE model_id = %s
            """
            
            return self.connector.execute_query(query, (model_id,))
        
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False

    def search_models(self, search_term: str) -> List[Dict[str, Any]]:
        """
        Search for models in the database.

        Args:
            search_term: The search term.

        Returns:
            List[Dict[str, Any]]: The matching models.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return []

        try:
            # Search for models
            query = """
                SELECT model_id, name, filename, surface_area, volume, centerline_length, 
                num_branch_points, num_endpoints, num_segments, created_at, updated_at
                FROM models
                WHERE model_id LIKE %s OR name LIKE %s OR filename LIKE %s
                ORDER BY updated_at DESC
            """
            
            search_pattern = f"%{search_term}%"
            models_data = self.connector.fetch_all(query, (search_pattern, search_pattern, search_pattern))
            
            models = []
            
            for model_data in models_data:
                # Create the model
                model = {
                    "model_id": model_data[0],
                    "name": model_data[1],
                    "filename": model_data[2],
                    "surface_area": model_data[3],
                    "volume": model_data[4],
                    "centerline_length": model_data[5],
                    "num_branch_points": model_data[6],
                    "num_endpoints": model_data[7],
                    "num_segments": model_data[8],
                    "created_at": model_data[9],
                    "updated_at": model_data[10]
                }
                
                models.append(model)
            
            return models
        
        except Exception as e:
            print(f"Error searching models: {e}")
            return []

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the models in the database.

        Returns:
            Dict[str, Any]: The statistics.
        """
        if not self.connector.is_connected():
            print("Error: Not connected to the database.")
            return {}

        try:
            # Get the statistics
            query = """
                SELECT 
                    COUNT(*) as num_models,
                    AVG(surface_area) as avg_surface_area,
                    AVG(volume) as avg_volume,
                    AVG(centerline_length) as avg_centerline_length,
                    AVG(num_branch_points) as avg_num_branch_points,
                    AVG(num_endpoints) as avg_num_endpoints,
                    AVG(num_segments) as avg_num_segments
                FROM models
            """
            
            stats_data = self.connector.fetch_one(query)
            
            if not stats_data:
                return {}
            
            # Create the statistics
            statistics = {
                "num_models": stats_data[0],
                "avg_surface_area": stats_data[1],
                "avg_volume": stats_data[2],
                "avg_centerline_length": stats_data[3],
                "avg_num_branch_points": stats_data[4],
                "avg_num_endpoints": stats_data[5],
                "avg_num_segments": stats_data[6]
            }
            
            return statistics
        
        except Exception as e:
            print(f"Error getting model statistics: {e}")
            return {}
