"""
STL Reader module for the blood vessel analysis system.

This module provides functionality for reading and processing STL blood vessel models.
"""

import os
import numpy as np
from stl import mesh
import trimesh

class STLReader:
    """Class for reading and processing STL blood vessel models."""
    
    def __init__(self):
        """Initialize the STL reader."""
        self.mesh = None
        self.trimesh = None
    
    def read_file(self, filename):
        """
        Read an STL file and store the mesh.
        
        Args:
            filename: Path to the STL file.
            
        Returns:
            bool: True if the file was read successfully, False otherwise.
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return False
        
        try:
            # Load the STL file using numpy-stl
            self.mesh = mesh.Mesh.from_file(filename)
            
            # Also load the file using trimesh for additional functionality
            self.trimesh = trimesh.load_mesh(filename)
            
            return True
        except Exception as e:
            print(f"Error reading STL file: {e}")
            return False
    
    def get_mesh_info(self):
        """
        Get basic information about the loaded mesh.
        
        Returns:
            dict: Dictionary containing mesh information.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return {}
        
        info = {
            "num_vertices": len(self.mesh.vectors.reshape(-1, 3)),
            "num_faces": len(self.mesh.vectors),
            "surface_area": self.get_surface_area(),
            "volume": self.get_volume(),
            "bounding_box": self.get_bounding_box()
        }
        
        return info
    
    def get_bounding_box(self):
        """
        Get the bounding box of the mesh.
        
        Returns:
            tuple: Tuple of minimum and maximum points of the bounding box.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return None
        
        vertices = self.mesh.vectors.reshape(-1, 3)
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        
        return (min_point, max_point)
    
    def get_surface_area(self):
        """
        Calculate the surface area of the mesh.
        
        Returns:
            float: Surface area of the mesh.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return 0.0
        
        return self.mesh.areas.sum()
    
    def get_volume(self):
        """
        Calculate the volume of the mesh.
        
        Returns:
            float: Volume of the mesh.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return 0.0
        
        if self.trimesh is not None and self.trimesh.is_watertight:
            return self.trimesh.volume
        
        # Fallback to numpy-stl volume calculation
        return abs(self.mesh.get_mass_properties()[0])
    
    def validate_mesh(self):
        """
        Validate the mesh for common issues.
        
        Returns:
            dict: Dictionary containing validation results.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return {}
        
        validation = {
            "is_watertight": False,
            "has_duplicate_faces": False,
            "has_degenerate_faces": False
        }
        
        if self.trimesh is not None:
            validation["is_watertight"] = self.trimesh.is_watertight
            validation["has_duplicate_faces"] = len(self.trimesh.duplicated_faces) > 0
            validation["has_degenerate_faces"] = len(self.trimesh.degenerate_faces) > 0
        
        return validation
    
    def repair_mesh(self):
        """
        Attempt to repair common mesh issues.
        
        Returns:
            bool: True if repairs were made, False otherwise.
        """
        if self.trimesh is None:
            print("Error: No trimesh loaded.")
            return False
        
        # Check if repairs are needed
        validation = self.validate_mesh()
        if validation.get("is_watertight", True) and not validation.get("has_duplicate_faces", False) and not validation.get("has_degenerate_faces", False):
            return False
        
        # Repair the mesh
        self.trimesh.fill_holes()
        self.trimesh.remove_duplicate_faces()
        self.trimesh.remove_degenerate_faces()
        
        # Update the numpy-stl mesh from the repaired trimesh
        self.mesh = mesh.Mesh(np.zeros(len(self.trimesh.faces), dtype=mesh.Mesh.dtype))
        for i, face in enumerate(self.trimesh.faces):
            for j in range(3):
                self.mesh.vectors[i][j] = self.trimesh.vertices[face[j]]
        
        return True
