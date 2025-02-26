"""
Core module for blood vessel STL analysis.

This module provides the main functionality for analyzing blood vessel STL models.
"""

import os
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from src.stl_processing.stl_reader import STLReader
from src.geometric_analysis.centerline import CenterlineExtractor
from src.geometric_analysis.cross_section import CrossSectionAnalyzer
from src.topology.vessel_network import VesselNetwork
from src.utils.data_structures import VesselModel, VesselSegment, VesselNode


class BloodVesselAnalyzer:
    """Main class for analyzing blood vessel STL models."""

    def __init__(self):
        """Initialize the blood vessel analyzer."""
        self.stl_reader = STLReader()
        self.centerline_extractor = CenterlineExtractor()
        self.cross_section_analyzer = CrossSectionAnalyzer()
        self.vessel_network = VesselNetwork()
        self.vessel_model = None

    def load_stl(self, filename: str) -> bool:
        """
        Load an STL file.

        Args:
            filename: Path to the STL file.

        Returns:
            bool: True if the file was loaded successfully, False otherwise.
        """
        success = self.stl_reader.read_file(filename)
        if success:
            # Create a new vessel model
            model_id = os.path.basename(filename).split('.')[0]
            self.vessel_model = VesselModel(model_id=model_id, name=os.path.basename(filename))
            
            # Store basic mesh information
            mesh_info = self.stl_reader.get_mesh_info()
            for key, value in mesh_info.items():
                self.vessel_model.set_property(f"mesh_{key}", value)
            
            # Store the filename
            self.vessel_model.set_property("filename", filename)
            
            return True
        else:
            return False

    def extract_centerline(self, method: str = 'skeleton') -> np.ndarray:
        """
        Extract the centerline from the loaded STL model.

        Args:
            method: The method to use for centerline extraction.

        Returns:
            np.ndarray: The centerline points.
        """
        if self.stl_reader.mesh is None:
            print("Error: No mesh loaded.")
            return np.array([])
        
        # Set the mesh for the centerline extractor
        self.centerline_extractor.set_mesh(self.stl_reader.mesh)
        
        # Extract the centerline
        centerline = self.centerline_extractor.extract_centerline(method=method)
        
        # Store the centerline in the vessel model
        if self.vessel_model is not None:
            self.vessel_model.set_property("centerline", centerline)
        
        return centerline

    def analyze_cross_sections(self) -> List[Dict[str, Any]]:
        """
        Analyze cross-sections of the loaded STL model.

        Returns:
            List[Dict[str, Any]]: List of dictionaries containing cross-section information.
        """
        if self.stl_reader.mesh is None:
            print("Error: No mesh loaded.")
            return [{"error": "No mesh loaded"}]
        
        # Get the centerline
        centerline = self.vessel_model.get_property("centerline")
        if centerline is None:
            centerline = self.extract_centerline()
        
        # Set up the cross-section analyzer
        self.cross_section_analyzer.set_mesh(self.stl_reader.mesh)
        self.cross_section_analyzer.set_centerline(centerline)
        
        # Compute cross-sections along the centerline
        cross_sections = self.cross_section_analyzer.compute_cross_sections_along_centerline(num_sections=20)
        
        # Analyze cross-section variation
        cross_section_analysis = self.cross_section_analyzer.analyze_cross_section_variation(cross_sections)
        
        # Store the results in the vessel model
        self.vessel_model.set_property("cross_sections", cross_sections)
        self.vessel_model.set_property("cross_section_analysis", cross_section_analysis)
        
        return cross_sections
        
    def analyze_geometry(self) -> Dict[str, Any]:
        """
        Analyze the geometry of the loaded STL model.

        Returns:
            Dict[str, Any]: Dictionary containing geometric analysis results.
        """
        if self.stl_reader.mesh is None:
            print("Error: No mesh loaded.")
            return {"error": "No mesh loaded"}
        
        if self.vessel_model is None:
            print("Error: No vessel model created.")
            return {"error": "No vessel model created"}
        
        # Get basic mesh information
        mesh_info = self.stl_reader.get_mesh_info()
        
        # Get the bounding box
        min_point, max_point = self.stl_reader.get_bounding_box()
        
        # Calculate the surface area
        surface_area = self.stl_reader.get_surface_area()
        
        # Validate the mesh
        validation = self.stl_reader.validate_mesh()
        
        # Extract the centerline if not already extracted
        centerline = self.vessel_model.get_property("centerline")
        if centerline is None:
            centerline = self.extract_centerline()
        
        # Get branch points and endpoints
        branch_points = self.centerline_extractor.get_branch_points()
        endpoints = self.centerline_extractor.get_endpoints()
        
        # Get centerline segments
        segments = self.centerline_extractor.get_centerline_segments()
        
        # Calculate segment lengths
        segment_lengths = self.centerline_extractor.calculate_segment_lengths()
        
        # Calculate segment diameters
        segment_diameters = self.centerline_extractor.calculate_segment_diameters()
        
        # Build the vessel network
        self.vessel_network.build_from_centerline(centerline, branch_points, endpoints, segments)
        
        # Get topological features
        topological_features = self.vessel_network.get_topological_features()
        
        # Calculate bifurcation angles
        bifurcation_angles = self.vessel_network.get_bifurcation_angles()
        
        # Calculate tortuosities
        tortuosities = self.vessel_network.calculate_all_tortuosities()
        
        # Set up the cross-section analyzer
        self.cross_section_analyzer.set_mesh(self.stl_reader.mesh)
        self.cross_section_analyzer.set_centerline(centerline)
        
        # Compute cross-sections along the centerline
        cross_sections = self.cross_section_analyzer.compute_cross_sections_along_centerline(num_sections=20)
        
        # Analyze cross-section variation
        cross_section_analysis = self.cross_section_analyzer.analyze_cross_section_variation(cross_sections)
        
        # Store the results in the vessel model
        self.vessel_model.set_property("surface_area", surface_area)
        self.vessel_model.set_property("bounding_box_min", min_point)
        self.vessel_model.set_property("bounding_box_max", max_point)
        self.vessel_model.set_property("validation", validation)
        self.vessel_model.set_property("branch_points", branch_points)
        self.vessel_model.set_property("endpoints", endpoints)
        self.vessel_model.set_property("segments", segments)
        self.vessel_model.set_property("segment_lengths", segment_lengths)
        self.vessel_model.set_property("segment_diameters", segment_diameters)
        self.vessel_model.set_property("topological_features", topological_features)
        self.vessel_model.set_property("bifurcation_angles", {k: v for k, v in bifurcation_angles.items()})
        self.vessel_model.set_property("tortuosities", tortuosities)
        self.vessel_model.set_property("cross_sections", cross_sections)
        self.vessel_model.set_property("cross_section_analysis", cross_section_analysis)
        
        # Create a summary of the results
        summary = {
            "mesh_info": mesh_info,
            "bounding_box": {
                "min": min_point.tolist() if isinstance(min_point, np.ndarray) else min_point,
                "max": max_point.tolist() if isinstance(max_point, np.ndarray) else max_point
            },
            "surface_area": surface_area,
            "validation": validation,
            "centerline_length": sum(segment_lengths),
            "num_branch_points": len(branch_points),
            "num_endpoints": len(endpoints),
            "num_segments": len(segments),
            "topological_features": topological_features,
            "cross_section_analysis": cross_section_analysis
        }
        
        return summary

    def build_vessel_model(self) -> VesselModel:
        """
        Build a complete vessel model from the analyzed data.

        Returns:
            VesselModel: The built vessel model.
        """
        if self.vessel_model is None:
            print("Error: No vessel model created.")
            return None
        
        # Get the centerline data
        centerline = self.vessel_model.get_property("centerline")
        branch_points = self.vessel_model.get_property("branch_points")
        endpoints = self.vessel_model.get_property("endpoints")
        segments = self.vessel_model.get_property("segments")
        segment_lengths = self.vessel_model.get_property("segment_lengths")
        segment_diameters = self.vessel_model.get_property("segment_diameters")
        
        if centerline is None:
            print("Error: Centerline data not available.")
            return self.vessel_model
            
        # If no segments are available, create a default segment for the test tube
        if segments is None or len(segments) == 0:
            segments = [centerline]
            self.vessel_model.set_property("segments", segments)
            
        # If no endpoints are available, use the first and last points of the centerline
        if endpoints is None or len(endpoints) == 0:
            endpoints = np.array([centerline[0], centerline[-1]])
            self.vessel_model.set_property("endpoints", endpoints)
            
        # If no branch points are available, use an empty array
        if branch_points is None:
            branch_points = np.array([])
            self.vessel_model.set_property("branch_points", branch_points)
        
        # Create nodes for branch points
        for i, point in enumerate(branch_points):
            node_id = f"B{i}"
            node = VesselNode(node_id=node_id, position=point, node_type="branch")
            self.vessel_model.add_node(node)
        
        # Create nodes for endpoints
        for i, point in enumerate(endpoints):
            node_id = f"E{i}"
            node = VesselNode(node_id=node_id, position=point, node_type="endpoint")
            self.vessel_model.add_node(node)
        
        # Create segments
        for i, segment_points in enumerate(segments):
            # Find the closest branch point or endpoint to the start of the segment
            start_point = segment_points[0]
            end_point = segment_points[-1]
            
            # Find the closest nodes
            start_node_id = None
            end_node_id = None
            
            # Check branch points
            for j, point in enumerate(branch_points):
                if np.linalg.norm(point - start_point) < 1e-6:
                    start_node_id = f"B{j}"
                if np.linalg.norm(point - end_point) < 1e-6:
                    end_node_id = f"B{j}"
            
            # Check endpoints
            for j, point in enumerate(endpoints):
                if np.linalg.norm(point - start_point) < 1e-6:
                    start_node_id = f"E{j}"
                if np.linalg.norm(point - end_point) < 1e-6:
                    end_node_id = f"E{j}"
            
            # Create the segment
            segment = VesselSegment(
                segment_id=i,
                points=segment_points,
                start_node_id=start_node_id,
                end_node_id=end_node_id
            )
            
            # Set segment properties
            if segment_lengths is not None and i < len(segment_lengths):
                segment.set_property("length", segment_lengths[i])
            
            if segment_diameters is not None and i < len(segment_diameters):
                segment.set_property("diameter", segment_diameters[i])
            
            # Add the segment to the model
            self.vessel_model.add_segment(segment)
        
        return self.vessel_model

    def save_model_to_json(self, filename: str):
        """
        Save the vessel model to a JSON file.

        Args:
            filename: Path to the output JSON file.
        """
        if self.vessel_model is None:
            print("Error: No vessel model created.")
            return
        
        self.vessel_model.save_to_json(filename)

    def load_model_from_json(self, filename: str) -> VesselModel:
        """
        Load a vessel model from a JSON file.

        Args:
            filename: Path to the input JSON file.

        Returns:
            VesselModel: The loaded vessel model.
        """
        self.vessel_model = VesselModel.load_from_json(filename)
        return self.vessel_model

    def export_results(self, output_dir: str):
        """
        Export analysis results to various formats.

        Args:
            output_dir: Directory to save the output files.
        """
        if self.vessel_model is None:
            print("Error: No vessel model created.")
            return
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the vessel model to JSON
        model_filename = os.path.join(output_dir, f"{self.vessel_model.model_id}_model.json")
        self.save_model_to_json(model_filename)
        
        # Export centerline data to CSV
        centerline = self.vessel_model.get_property("centerline")
        if centerline is not None:
            centerline_filename = os.path.join(output_dir, f"{self.vessel_model.model_id}_centerline.csv")
            np.savetxt(centerline_filename, centerline, delimiter=',', header='x,y,z', comments='')
        
        # Export cross-section data to CSV
        cross_sections = self.vessel_model.get_property("cross_sections")
        if cross_sections is not None:
            cross_section_filename = os.path.join(output_dir, f"{self.vessel_model.model_id}_cross_sections.csv")
            with open(cross_section_filename, 'w') as f:
                f.write("index,position_x,position_y,position_z,area,perimeter,equivalent_diameter,circularity\n")
                for cs in cross_sections:
                    if "error" in cs:
                        continue
                    
                    position = cs.get("position", [0, 0, 0])
                    f.write(f"{cs.get('index', 0)},{position[0]},{position[1]},{position[2]},{cs.get('area', 0)},"
                            f"{cs.get('perimeter', 0)},{cs.get('equivalent_diameter', 0)},{cs.get('circularity', 0)}\n")
        
        # Export segment data to CSV
        segments = self.vessel_model.get_all_segments()
        if segments:
            segment_filename = os.path.join(output_dir, f"{self.vessel_model.model_id}_segments.csv")
            with open(segment_filename, 'w') as f:
                f.write("segment_id,start_node_id,end_node_id,length,diameter\n")
                for segment_id, segment in segments.items():
                    length = segment.get_property("length", 0)
                    diameter = segment.get_property("diameter", 0)
                    f.write(f"{segment_id},{segment.start_node_id},{segment.end_node_id},{length},{diameter}\n")
