"""
Centerline Extractor module for the blood vessel analysis system.

This module provides functionality for extracting centerlines from blood vessel models.
"""

import numpy as np
from scipy import ndimage
import networkx as nx

class CenterlineExtractor:
    """Class for extracting centerlines from blood vessel models."""
    
    def __init__(self, mesh=None):
        """
        Initialize the centerline extractor.
        
        Args:
            mesh: The mesh to extract centerlines from.
        """
        self.mesh = mesh
        self.centerline = None
        self.centerline_graph = None
        self.branch_points = None
        self.endpoints = None
    
    def set_mesh(self, mesh):
        """
        Set the mesh to extract centerlines from.
        
        Args:
            mesh: The mesh to extract centerlines from.
        """
        self.mesh = mesh
        self.centerline = None
        self.centerline_graph = None
        self.branch_points = None
        self.endpoints = None
    
    def extract_centerline(self, method='skeleton'):
        """
        Extract the centerline from the mesh.
        
        Args:
            method: The method to use for centerline extraction ('skeleton' or 'medial_axis').
            
        Returns:
            numpy.ndarray: Array of centerline points.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return None
        
        if method == 'skeleton':
            self.centerline = self._extract_skeleton()
        elif method == 'medial_axis':
            self.centerline = self._extract_medial_axis()
        else:
            print(f"Error: Unknown method '{method}'. Using 'skeleton' instead.")
            self.centerline = self._extract_skeleton()
        
        # Build the centerline graph
        self._build_centerline_graph()
        
        # Extract branch points and endpoints
        self._extract_branch_points_and_endpoints()
        
        return self.centerline
    
    def _extract_skeleton(self):
        """
        Extract the centerline using skeletonization.
        
        Returns:
            numpy.ndarray: Array of centerline points.
        """
        # For demonstration purposes, we'll create a simple centerline
        # In a real implementation, this would use skeletonization algorithms
        
        # Get the bounding box
        vertices = self.mesh.vectors.reshape(-1, 3)
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        
        # Create a centerline along the longest dimension
        longest_dim = np.argmax(max_point - min_point)
        other_dims = [i for i in range(3) if i != longest_dim]
        
        # Create points along the centerline
        num_points = 20
        centerline_points = []
        
        for i in range(num_points):
            point = np.zeros(3)
            t = i / (num_points - 1)
            point[longest_dim] = min_point[longest_dim] + t * (max_point[longest_dim] - min_point[longest_dim])
            point[other_dims[0]] = (min_point[other_dims[0]] + max_point[other_dims[0]]) / 2
            point[other_dims[1]] = (min_point[other_dims[1]] + max_point[other_dims[1]]) / 2
            centerline_points.append(point)
        
        return np.array(centerline_points)
    
    def _extract_medial_axis(self):
        """
        Extract the centerline using medial axis transform.
        
        Returns:
            numpy.ndarray: Array of centerline points.
        """
        # For demonstration purposes, we'll create a simple centerline
        # In a real implementation, this would use medial axis transform
        
        # Get the bounding box
        vertices = self.mesh.vectors.reshape(-1, 3)
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        
        # Create a centerline along the longest dimension
        longest_dim = np.argmax(max_point - min_point)
        other_dims = [i for i in range(3) if i != longest_dim]
        
        # Create points along the centerline
        num_points = 20
        centerline_points = []
        
        for i in range(num_points):
            point = np.zeros(3)
            t = i / (num_points - 1)
            point[longest_dim] = min_point[longest_dim] + t * (max_point[longest_dim] - min_point[longest_dim])
            point[other_dims[0]] = (min_point[other_dims[0]] + max_point[other_dims[0]]) / 2
            point[other_dims[1]] = (min_point[other_dims[1]] + max_point[other_dims[1]]) / 2
            centerline_points.append(point)
        
        return np.array(centerline_points)
    
    def _build_centerline_graph(self):
        """Build a graph representation of the centerline."""
        if self.centerline is None:
            print("Error: No centerline extracted.")
            return
        
        # Create a graph
        self.centerline_graph = nx.Graph()
        
        # Add nodes for each centerline point
        for i, point in enumerate(self.centerline):
            self.centerline_graph.add_node(i, position=point)
        
        # Add edges between consecutive points
        for i in range(len(self.centerline) - 1):
            self.centerline_graph.add_edge(i, i + 1)
    
    def _extract_branch_points_and_endpoints(self):
        """Extract branch points and endpoints from the centerline graph."""
        if self.centerline_graph is None:
            print("Error: No centerline graph built.")
            return
        
        # Find branch points (nodes with degree > 2)
        branch_point_indices = [node for node, degree in self.centerline_graph.degree() if degree > 2]
        self.branch_points = np.array([self.centerline_graph.nodes[i]['position'] for i in branch_point_indices])
        
        # Find endpoints (nodes with degree == 1)
        endpoint_indices = [node for node, degree in self.centerline_graph.degree() if degree == 1]
        self.endpoints = np.array([self.centerline_graph.nodes[i]['position'] for i in endpoint_indices])
    
    def get_centerline_graph(self):
        """
        Get the centerline graph.
        
        Returns:
            networkx.Graph: Graph of the centerline, or None if not computed.
        """
        return self.centerline_graph
    
    def get_branch_points(self):
        """
        Get the branch points of the centerline.
        
        Returns:
            numpy.ndarray: Array of branch point coordinates.
        """
        if self.branch_points is None or len(self.branch_points) == 0:
            # For demonstration, return an empty array
            return np.array([])
        return self.branch_points
    
    def get_endpoints(self):
        """
        Get the endpoints of the centerline.
        
        Returns:
            numpy.ndarray: Array of endpoint coordinates.
        """
        if self.endpoints is None or len(self.endpoints) == 0:
            # For demonstration, return the first and last points of the centerline
            if self.centerline is not None and len(self.centerline) >= 2:
                return np.array([self.centerline[0], self.centerline[-1]])
            return np.array([])
        return self.endpoints
    
    def get_centerline_segments(self):
        """
        Get the centerline segments.
        
        Returns:
            list: List of numpy arrays, each representing a centerline segment.
        """
        if self.centerline is None:
            print("Error: No centerline extracted.")
            return []
        
        # For demonstration, return a single segment
        return [self.centerline]
    
    def calculate_segment_lengths(self):
        """
        Calculate the length of each centerline segment.
        
        Returns:
            list: List of segment lengths.
        """
        segments = self.get_centerline_segments()
        lengths = []
        
        for segment in segments:
            length = 0.0
            for i in range(len(segment) - 1):
                length += np.linalg.norm(segment[i+1] - segment[i])
            lengths.append(length)
        
        return lengths
    
    def calculate_segment_diameters(self):
        """
        Estimate the diameter of each centerline segment.
        
        Returns:
            list: List of segment diameters.
        """
        if self.mesh is None or self.centerline is None:
            print("Error: No mesh or centerline loaded.")
            return []
        
        # For demonstration, return a constant diameter
        segments = self.get_centerline_segments()
        return [1.0] * len(segments)
