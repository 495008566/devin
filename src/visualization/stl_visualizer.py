"""
STL Visualizer module for the blood vessel analysis system.

This module provides functionality for visualizing STL blood vessel models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

class STLVisualizer:
    """Class for visualizing STL blood vessel models."""
    
    def __init__(self):
        """Initialize the STL visualizer."""
        self.mesh = None
    
    def load_stl(self, filename):
        """
        Load an STL file.
        
        Args:
            filename: Path to the STL file.
            
        Returns:
            bool: True if the file was loaded successfully, False otherwise.
        """
        if not os.path.exists(filename):
            print(f"Error: File {filename} does not exist.")
            return False
        
        try:
            # Load the STL file using numpy-stl
            self.mesh = mesh.Mesh.from_file(filename)
            return True
        except Exception as e:
            print(f"Error loading STL file: {e}")
            return False
    
    def save_visualization(self, output_file):
        """
        Save a visualization of the STL file to an image file.
        
        Args:
            output_file: Path to save the visualization.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a collection of triangles
        triangles = self.mesh.vectors
        
        # Create a Poly3DCollection
        collection = Poly3DCollection(triangles, alpha=0.7, edgecolor='k', linewidth=0.1)
        collection.set_facecolor('red')
        
        # Add the collection to the plot
        ax.add_collection3d(collection)
        
        # Auto-scale the axes
        scale = self.mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Blood Vessel STL Model')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
    
    def save_visualization_with_centerline(self, output_file, centerline):
        """
        Save a visualization of the STL file with centerline to an image file.
        
        Args:
            output_file: Path to save the visualization.
            centerline: Centerline points.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a collection of triangles
        triangles = self.mesh.vectors
        
        # Create a Poly3DCollection
        collection = Poly3DCollection(triangles, alpha=0.5, edgecolor='k', linewidth=0.1)
        collection.set_facecolor('red')
        
        # Add the collection to the plot
        ax.add_collection3d(collection)
        
        # Plot the centerline
        if centerline is not None:
            ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'b-', linewidth=2, label='Centerline')
            ax.scatter(centerline[:, 0], centerline[:, 1], centerline[:, 2], color='blue', s=20)
        
        # Auto-scale the axes
        scale = self.mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Blood Vessel STL Model with Centerline')
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
    
    def save_visualization_with_cross_sections(self, output_file, centerline, cross_sections):
        """
        Save a visualization of the STL file with cross-sections to an image file.
        
        Args:
            output_file: Path to save the visualization.
            centerline: Centerline points.
            cross_sections: Cross-section information.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a collection of triangles
        triangles = self.mesh.vectors
        
        # Create a Poly3DCollection
        collection = Poly3DCollection(triangles, alpha=0.3, edgecolor='k', linewidth=0.1)
        collection.set_facecolor('red')
        
        # Add the collection to the plot
        ax.add_collection3d(collection)
        
        # Plot the centerline
        if centerline is not None:
            ax.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], 'b-', linewidth=2, label='Centerline')
        
        # Plot cross-section points
        if cross_sections is not None:
            num_sections = len(cross_sections)
            for i, cs in enumerate(cross_sections):
                # Get a point along the centerline
                idx = int(i * (len(centerline) - 1) / (num_sections - 1))
                point = centerline[idx]
                
                # Get the direction at this point
                if idx < len(centerline) - 1:
                    direction = centerline[idx + 1] - centerline[idx]
                else:
                    direction = centerline[idx] - centerline[idx - 1]
                
                direction = direction / np.linalg.norm(direction)
                
                # Plot the cross-section point
                ax.scatter(point[0], point[1], point[2], color='green', s=50, label='Cross-section' if i == 0 else "")
                
                # Add a text label with the area
                if 'area' in cs:
                    ax.text(point[0], point[1], point[2], f"A={cs['area']:.2f}", color='black')
        
        # Auto-scale the axes
        scale = self.mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Blood Vessel STL Model with Cross-Sections')
        
        # Add a legend (only for the first items to avoid duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
    
    def save_visualization_with_topology(self, output_file, model):
        """
        Save a visualization of the vessel topology to an image file.
        
        Args:
            output_file: Path to save the visualization.
            model: Vessel model.
        """
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return
        
        # Create a 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create a collection of triangles
        triangles = self.mesh.vectors
        
        # Create a Poly3DCollection
        collection = Poly3DCollection(triangles, alpha=0.2, edgecolor='k', linewidth=0.1)
        collection.set_facecolor('red')
        
        # Add the collection to the plot
        ax.add_collection3d(collection)
        
        # Plot the segments
        segments = model.get_all_segments()
        for segment_id, segment in segments.items():
            points = segment.points
            ax.plot(points[:, 0], points[:, 1], points[:, 2], 'b-', linewidth=2)
        
        # Plot the branch points
        branch_points = model.get_branch_points()
        if branch_points and len(branch_points) > 0:
            branch_points = np.array(branch_points)
            ax.scatter(branch_points[:, 0], branch_points[:, 1], branch_points[:, 2], 
                      color='green', s=100, label='Branch Points')
        
        # Plot the endpoints
        endpoints = model.get_endpoints()
        if endpoints and len(endpoints) > 0:
            endpoints = np.array(endpoints)
            ax.scatter(endpoints[:, 0], endpoints[:, 1], endpoints[:, 2], 
                      color='orange', s=100, label='Endpoints')
        
        # Auto-scale the axes
        scale = self.mesh.points.flatten()
        ax.auto_scale_xyz(scale, scale, scale)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Blood Vessel Topology')
        
        # Add a legend
        ax.legend()
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
