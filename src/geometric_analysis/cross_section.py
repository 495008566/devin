"""
Cross Section Analyzer module for the blood vessel analysis system.

This module provides functionality for analyzing cross-sections of blood vessel models.
"""

import numpy as np
from scipy import spatial

class CrossSectionAnalyzer:
    """Class for analyzing cross-sections of blood vessel models."""
    
    def __init__(self, mesh=None):
        """
        Initialize the cross-section analyzer.
        
        Args:
            mesh: The mesh to analyze.
        """
        self.mesh = mesh
    
    def set_mesh(self, mesh):
        """
        Set the mesh to analyze.
        
        Args:
            mesh: The mesh to analyze.
        """
        self.mesh = mesh
    
    def compute_cross_section(self, point, normal):
        """
        Compute a cross-section at the specified point with the given normal.
        
        Args:
            point: Point on the centerline.
            normal: Normal vector (direction of the cross-section).
            
        Returns:
            dict: Dictionary containing cross-section information.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return {"error": "No mesh loaded"}
        
        # Normalize the normal vector
        normal = normal / np.linalg.norm(normal)
        
        # For demonstration purposes, we'll create a simple cross-section
        # In a real implementation, this would compute the intersection of the mesh with a plane
        
        # Create a circular cross-section
        radius = 1.0  # Default radius
        
        # Calculate the area of the cross-section
        area = np.pi * radius**2
        
        # Calculate the perimeter of the cross-section
        perimeter = 2 * np.pi * radius
        
        # Calculate the equivalent diameter
        equivalent_diameter = 2 * radius
        
        # Calculate the circularity
        circularity = 4 * np.pi * area / (perimeter**2)
        
        # Return the cross-section information
        return {
            "point": point,
            "normal": normal,
            "area": area,
            "perimeter": perimeter,
            "equivalent_diameter": equivalent_diameter,
            "circularity": circularity
        }
    
    def compute_cross_sections_along_centerline(self, centerline, num_sections=10):
        """
        Compute cross-sections along the centerline.
        
        Args:
            centerline: Centerline points.
            num_sections: Number of cross-sections to compute.
            
        Returns:
            list: List of dictionaries containing cross-section information.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return [{"error": "No mesh loaded"}]
        
        if centerline is None or len(centerline) < 2:
            print("Error: Invalid centerline.")
            return [{"error": "Invalid centerline"}]
        
        # Compute cross-sections at evenly spaced points along the centerline
        cross_sections = []
        
        for i in range(num_sections):
            # Get a point along the centerline
            idx = int(i * (len(centerline) - 1) / (num_sections - 1))
            point = centerline[idx]
            
            # Get the direction at this point
            if idx < len(centerline) - 1:
                direction = centerline[idx + 1] - centerline[idx]
            else:
                direction = centerline[idx] - centerline[idx - 1]
            
            direction = direction / np.linalg.norm(direction)
            
            # Compute the cross-section
            cross_section = self.compute_cross_section(point, direction)
            cross_sections.append(cross_section)
        
        return cross_sections
