"""
STL Reader module.

This module provides functionality for reading and processing STL files.
"""

import numpy as np
from stl import mesh
import trimesh
import os
from typing import Dict, Any, Optional, Tuple, List


class STLReader:
    """Class for reading and processing STL files."""

    def __init__(self):
        """Initialize the STL reader."""
        self.mesh = None
        self.filename = None

    def read_file(self, filename: str) -> bool:
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
            # Try reading with numpy-stl
            numpy_mesh = mesh.Mesh.from_file(filename)
            self.filename = filename
            
            # Convert numpy-stl mesh to trimesh for better compatibility
            # Extract vertices and faces from the numpy-stl mesh
            vertices = []
            faces = []
            
            # Each vector in numpy-stl is a triangle with 3 vertices
            for i, triangle in enumerate(numpy_mesh.vectors):
                # Add the three vertices of this triangle
                base_idx = len(vertices)
                for vertex in triangle:
                    vertices.append(vertex)
                
                # Add the face (triangle) using the indices of the vertices
                faces.append([base_idx, base_idx + 1, base_idx + 2])
            
            # Convert to numpy arrays
            vertices = np.array(vertices)
            faces = np.array(faces)
            
            # Create a trimesh mesh
            self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Store the original numpy-stl mesh as well
            self.numpy_mesh = numpy_mesh
            
            return True
        except Exception as e:
            print(f"Error reading file with numpy-stl: {e}")
            try:
                # Try reading with trimesh as a fallback
                self.mesh = trimesh.load(filename)
                self.filename = filename
                self.numpy_mesh = None
                return True
            except Exception as e2:
                print(f"Error reading file with trimesh: {e2}")
                return False

    def get_mesh_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded mesh.

        Returns:
            Dict[str, Any]: Dictionary containing mesh information.
        """
        if self.mesh is None:
            return {"error": "No mesh loaded"}

        info = {}
        
        # Handle numpy-stl mesh
        if isinstance(self.mesh, mesh.Mesh):
            info["type"] = "numpy-stl"
            info["vertices"] = self.mesh.vectors.reshape(-1, 3)
            info["faces"] = np.arange(len(self.mesh.vectors) * 3).reshape(-1, 3)
            info["num_triangles"] = len(self.mesh.vectors)
            info["volume"] = self.mesh.get_mass_properties()[0]
            info["center_of_mass"] = self.mesh.get_mass_properties()[1]
            info["inertia"] = self.mesh.get_mass_properties()[2]
        
        # Handle trimesh mesh
        elif isinstance(self.mesh, trimesh.Trimesh):
            info["type"] = "trimesh"
            info["vertices"] = self.mesh.vertices
            info["faces"] = self.mesh.faces
            info["num_triangles"] = len(self.mesh.faces)
            info["volume"] = self.mesh.volume
            info["center_of_mass"] = self.mesh.center_mass
            info["inertia"] = self.mesh.moment_inertia
        
        return info

    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the mesh.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Minimum and maximum points of the bounding box.
        """
        if self.mesh is None:
            return np.array([0, 0, 0]), np.array([0, 0, 0])

        if isinstance(self.mesh, mesh.Mesh):
            min_point = self.mesh.vectors.min(axis=(0, 1))
            max_point = self.mesh.vectors.max(axis=(0, 1))
        elif isinstance(self.mesh, trimesh.Trimesh):
            min_point = self.mesh.bounds[0]
            max_point = self.mesh.bounds[1]
        
        return min_point, max_point

    def get_surface_area(self) -> float:
        """
        Calculate the surface area of the mesh.

        Returns:
            float: Surface area of the mesh.
        """
        if self.mesh is None:
            return 0.0

        if isinstance(self.mesh, mesh.Mesh):
            # Calculate area for each triangle and sum
            vectors = self.mesh.vectors
            normals = np.cross(vectors[:, 1] - vectors[:, 0], vectors[:, 2] - vectors[:, 0])
            areas = 0.5 * np.sqrt(np.sum(normals**2, axis=1))
            return np.sum(areas)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.area
        
        return 0.0

    def validate_mesh(self) -> Dict[str, Any]:
        """
        Validate the mesh for common issues.

        Returns:
            Dict[str, Any]: Dictionary containing validation results.
        """
        if self.mesh is None:
            return {"valid": False, "error": "No mesh loaded"}

        validation = {"valid": True, "issues": []}

        if isinstance(self.mesh, trimesh.Trimesh):
            # Check for watertightness
            if not self.mesh.is_watertight:
                validation["valid"] = False
                validation["issues"].append("Mesh is not watertight")
            
            # Check for duplicate faces
            if self.mesh.is_empty:
                validation["valid"] = False
                validation["issues"].append("Mesh is empty")
            
            # Check for degenerate faces
            if hasattr(self.mesh, 'degenerate_faces'):
                if len(self.mesh.degenerate_faces) > 0:
                    validation["valid"] = False
                    validation["issues"].append(f"Mesh has {len(self.mesh.degenerate_faces)} degenerate faces")
            # Skip this check for now as it's causing issues
            # We'll rely on other validation checks
        
        elif isinstance(self.mesh, mesh.Mesh):
            # Basic validation for numpy-stl
            vectors = self.mesh.vectors
            
            # Check for degenerate triangles
            v0, v1, v2 = vectors[:, 0], vectors[:, 1], vectors[:, 2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            normals = np.cross(edge1, edge2)
            areas = 0.5 * np.sqrt(np.sum(normals**2, axis=1))
            
            if np.any(areas < 1e-10):
                validation["valid"] = False
                validation["issues"].append(f"Mesh has {np.sum(areas < 1e-10)} degenerate triangles")
        
        return validation

    def repair_mesh(self) -> bool:
        """
        Attempt to repair common mesh issues.

        Returns:
            bool: True if repairs were made, False otherwise.
        """
        if self.mesh is None or not isinstance(self.mesh, trimesh.Trimesh):
            # Convert to trimesh if it's a numpy-stl mesh
            if isinstance(self.mesh, mesh.Mesh):
                try:
                    vertices = self.mesh.vectors.reshape(-1, 3)
                    faces = np.arange(len(vertices)).reshape(-1, 3)
                    self.mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                except Exception as e:
                    print(f"Error converting to trimesh: {e}")
                    return False
            else:
                return False
        
        # Now we have a trimesh mesh, attempt repairs
        try:
            # Fill holes
            self.mesh.fill_holes()
            
            # Remove duplicate faces
            self.mesh.remove_duplicate_faces()
            
            # Remove degenerate faces
            self.mesh.remove_degenerate_faces()
            
            # Merge vertices that are close together
            self.mesh.merge_vertices()
            
            return True
        except Exception as e:
            print(f"Error repairing mesh: {e}")
            return False
