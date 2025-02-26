"""
Core module for the blood vessel analysis system.

This module provides the main functionality for analyzing blood vessel models.
"""

import os
import numpy as np
import json

from src.stl_processing.stl_reader import STLReader
from src.geometric_analysis.centerline import CenterlineExtractor
from src.geometric_analysis.cross_section import CrossSectionAnalyzer
from src.topology.vessel_network import VesselNetwork
from src.utils.data_structures import VesselModel, VesselNode, VesselSegment

class BloodVesselAnalyzer:
    """Class for analyzing blood vessel models."""
    
    def __init__(self):
        """Initialize the blood vessel analyzer."""
        self.reader = STLReader()
        self.centerline_extractor = CenterlineExtractor()
        self.cross_section_analyzer = CrossSectionAnalyzer()
        self.vessel_network = VesselNetwork()
        self.model = None
    
    def load_stl(self, filename):
        """
        Load an STL file.
        
        Args:
            filename: Path to the STL file.
            
        Returns:
            bool: True if the file was loaded successfully, False otherwise.
        """
        success = self.reader.read_file(filename)
        if success:
            self.centerline_extractor.set_mesh(self.reader.mesh)
            self.cross_section_analyzer.set_mesh(self.reader.mesh)
            self.model = VesselModel(model_id=os.path.basename(filename))
            self.model.set_property("filename", filename)
            self.model.set_property("mesh_info", self.reader.get_mesh_info())
        return success
    
    def extract_centerline(self, method='skeleton'):
        """
        Extract the centerline from the loaded mesh.
        
        Args:
            method: The method to use for centerline extraction ('skeleton' or 'medial_axis').
            
        Returns:
            numpy.ndarray: Array of centerline points.
        """
        centerline = self.centerline_extractor.extract_centerline(method)
        if centerline is not None:
            self.vessel_network.set_centerline(centerline)
            self.vessel_network.set_branch_points(self.centerline_extractor.get_branch_points())
            self.vessel_network.set_endpoints(self.centerline_extractor.get_endpoints())
        return centerline
    
    def analyze_cross_sections(self, centerline, num_sections=10):
        """
        Analyze cross-sections along the centerline.
        
        Args:
            centerline: Centerline points.
            num_sections: Number of cross-sections to analyze.
            
        Returns:
            list: List of dictionaries containing cross-section information.
        """
        return self.cross_section_analyzer.compute_cross_sections_along_centerline(centerline, num_sections)
    
    def analyze_geometry(self):
        """
        Analyze the geometry of the loaded mesh.
        
        Returns:
            dict: Dictionary containing geometric information.
        """
        if self.reader.mesh is None:
            print("Error: No mesh loaded.")
            return {}
        
        # Get basic mesh information
        mesh_info = self.reader.get_mesh_info()
        
        # Extract the centerline if not already extracted
        if self.vessel_network.centerline is None:
            self.extract_centerline()
        
        # Calculate centerline segment lengths
        segment_lengths = self.centerline_extractor.calculate_segment_lengths()
        
        # Calculate centerline segment diameters
        segment_diameters = self.centerline_extractor.calculate_segment_diameters()
        
        # Build the vessel network
        self.vessel_network.build_network()
        
        # Calculate bifurcation angles
        bifurcation_angles = self.vessel_network.get_bifurcation_angles()
        
        # Combine all information
        geometry_info = {
            "surface_area": mesh_info.get("surface_area", 0.0),
            "volume": mesh_info.get("volume", 0.0),
            "num_vertices": mesh_info.get("num_vertices", 0),
            "num_faces": mesh_info.get("num_faces", 0),
            "segment_lengths": segment_lengths,
            "segment_diameters": segment_diameters,
            "num_segments": len(segment_lengths),
            "total_length": sum(segment_lengths),
            "average_diameter": sum(segment_diameters) / len(segment_diameters) if segment_diameters else 0.0,
            "num_branch_points": len(self.centerline_extractor.get_branch_points()),
            "num_endpoints": len(self.centerline_extractor.get_endpoints()),
            "bifurcation_angles": bifurcation_angles,
            "average_bifurcation_angle": sum(bifurcation_angles) / len(bifurcation_angles) if bifurcation_angles else 0.0
        }
        
        return geometry_info
    
    def build_vessel_model(self):
        """
        Build a vessel model from the analyzed data.
        
        Returns:
            VesselModel: The built vessel model.
        """
        if self.model is None:
            self.model = VesselModel()
        
        # Extract the centerline if not already extracted
        if self.vessel_network.centerline is None:
            self.extract_centerline()
        
        # Add branch points as nodes
        branch_points = self.centerline_extractor.get_branch_points()
        for i, point in enumerate(branch_points):
            node_id = f"branch_{i}"
            node = VesselNode(node_id, point, "branch")
            self.model.add_node(node)
        
        # Add endpoints as nodes
        endpoints = self.centerline_extractor.get_endpoints()
        for i, point in enumerate(endpoints):
            node_id = f"endpoint_{i}"
            node = VesselNode(node_id, point, "endpoint")
            self.model.add_node(node)
        
        # Add centerline segments
        segments = self.centerline_extractor.get_centerline_segments()
        for i, segment_points in enumerate(segments):
            segment_id = f"segment_{i}"
            
            # Find the closest branch point or endpoint to the start of the segment
            start_node_id = None
            min_distance = float('inf')
            for node_id, node in self.model.get_all_nodes().items():
                distance = np.linalg.norm(node.position - segment_points[0])
                if distance < min_distance:
                    min_distance = distance
                    start_node_id = node_id
            
            # Find the closest branch point or endpoint to the end of the segment
            end_node_id = None
            min_distance = float('inf')
            for node_id, node in self.model.get_all_nodes().items():
                distance = np.linalg.norm(node.position - segment_points[-1])
                if distance < min_distance:
                    min_distance = distance
                    end_node_id = node_id
            
            # Create the segment
            segment = VesselSegment(segment_id, segment_points, start_node_id, end_node_id)
            
            # Calculate segment properties
            length = segment.calculate_length()
            segment.set_property("length", length)
            
            # Add the segment to the model
            self.model.add_segment(segment)
            
            # Update the connected segments of the nodes
            if start_node_id is not None:
                start_node = self.model.get_node(start_node_id)
                if start_node is not None:
                    start_node.add_connected_segment(segment_id)
            
            if end_node_id is not None:
                end_node = self.model.get_node(end_node_id)
                if end_node is not None:
                    end_node.add_connected_segment(segment_id)
        
        # Add geometric information to the model
        geometry_info = self.analyze_geometry()
        for key, value in geometry_info.items():
            self.model.set_property(key, value)
        
        return self.model
    
    def save_model_to_json(self, filename):
        """
        Save the vessel model to a JSON file.
        
        Args:
            filename: Path to save the JSON file.
            
        Returns:
            bool: True if the file was saved successfully, False otherwise.
        """
        if self.model is None:
            print("Error: No model built.")
            return False
        
        return self.model.save_to_file(filename)
