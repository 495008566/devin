"""
Vessel Network module for the blood vessel analysis system.

This module provides functionality for analyzing vessel network topology.
"""

import numpy as np
import networkx as nx

class VesselNetwork:
    """Class for analyzing vessel network topology."""
    
    def __init__(self, centerline=None, branch_points=None, endpoints=None):
        """
        Initialize the vessel network.
        
        Args:
            centerline: Centerline points.
            branch_points: Branch point coordinates.
            endpoints: Endpoint coordinates.
        """
        self.centerline = centerline
        self.branch_points = branch_points
        self.endpoints = endpoints
        self.network = None
    
    def set_centerline(self, centerline):
        """
        Set the centerline.
        
        Args:
            centerline: Centerline points.
        """
        self.centerline = centerline
    
    def set_branch_points(self, branch_points):
        """
        Set the branch points.
        
        Args:
            branch_points: Branch point coordinates.
        """
        self.branch_points = branch_points
    
    def set_endpoints(self, endpoints):
        """
        Set the endpoints.
        
        Args:
            endpoints: Endpoint coordinates.
        """
        self.endpoints = endpoints
    
    def build_network(self):
        """
        Build the vessel network graph.
        
        Returns:
            networkx.Graph: Graph of the vessel network.
        """
        if self.centerline is None:
            print("Error: No centerline provided.")
            return None
        
        # Create a graph
        self.network = nx.Graph()
        
        # Add nodes for each centerline point
        for i, point in enumerate(self.centerline):
            self.network.add_node(i, position=point)
        
        # Add edges between consecutive points
        for i in range(len(self.centerline) - 1):
            self.network.add_edge(i, i + 1)
        
        # If branch points are provided, add them to the graph
        if self.branch_points is not None and len(self.branch_points) > 0:
            for i, point in enumerate(self.branch_points):
                node_id = len(self.centerline) + i
                self.network.add_node(node_id, position=point, type='branch')
                
                # Find the closest centerline points and connect them
                for j, centerline_point in enumerate(self.centerline):
                    if np.linalg.norm(point - centerline_point) < 0.1:
                        self.network.add_edge(node_id, j)
        
        # If endpoints are provided, add them to the graph
        if self.endpoints is not None and len(self.endpoints) > 0:
            for i, point in enumerate(self.endpoints):
                node_id = len(self.centerline) + len(self.branch_points) + i
                self.network.add_node(node_id, position=point, type='endpoint')
                
                # Find the closest centerline points and connect them
                for j, centerline_point in enumerate(self.centerline):
                    if np.linalg.norm(point - centerline_point) < 0.1:
                        self.network.add_edge(node_id, j)
        
        return self.network
    
    def get_bifurcation_angles(self):
        """
        Calculate the bifurcation angles.
        
        Returns:
            list: List of bifurcation angles in degrees.
        """
        if self.network is None:
            print("Error: No network built.")
            return []
        
        if self.branch_points is None or len(self.branch_points) == 0:
            print("Error: No branch points provided.")
            return []
        
        angles = []
        
        # For each branch point, calculate the angles between its connected segments
        for i, point in enumerate(self.branch_points):
            node_id = len(self.centerline) + i
            
            # Get the neighbors of the branch point
            neighbors = list(self.network.neighbors(node_id))
            
            if len(neighbors) >= 3:
                # Calculate the angles between each pair of segments
                for j in range(len(neighbors)):
                    for k in range(j + 1, len(neighbors)):
                        # Get the positions of the neighbors
                        pos_j = self.network.nodes[neighbors[j]]['position']
                        pos_k = self.network.nodes[neighbors[k]]['position']
                        
                        # Calculate the vectors from the branch point to the neighbors
                        vec_j = pos_j - point
                        vec_k = pos_k - point
                        
                        # Normalize the vectors
                        vec_j = vec_j / np.linalg.norm(vec_j)
                        vec_k = vec_k / np.linalg.norm(vec_k)
                        
                        # Calculate the angle between the vectors
                        cos_angle = np.dot(vec_j, vec_k)
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        
                        # Convert to degrees
                        angle_deg = np.degrees(angle)
                        
                        angles.append(angle_deg)
        
        return angles
    
    def get_segment_connectivity(self):
        """
        Get the connectivity of vessel segments.
        
        Returns:
            dict: Dictionary mapping segment IDs to lists of connected segment IDs.
        """
        if self.network is None:
            print("Error: No network built.")
            return {}
        
        # For demonstration purposes, we'll create a simple connectivity map
        # In a real implementation, this would analyze the network structure
        
        connectivity = {}
        
        # For each segment, find its connected segments
        for i in range(len(self.centerline) - 1):
            connectivity[i] = [i + 1]
        
        connectivity[len(self.centerline) - 1] = []
        
        return connectivity
