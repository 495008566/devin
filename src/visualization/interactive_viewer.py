"""
Interactive Viewer module for the blood vessel analysis system.

This module provides functionality for interactive viewing of STL blood vessel models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from stl import mesh

class InteractiveViewer:
    """Class for interactive viewing of STL blood vessel models."""
    
    def __init__(self):
        """Initialize the interactive viewer."""
        self.mesh = None
        self.fig = None
        self.ax = None
    
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
    
    def start_interactive_session(self):
        """Start an interactive viewing session."""
        if self.mesh is None:
            print("Error: No mesh loaded.")
            return
        
        # Create a 3D figure
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Plot the triangles
        for i in range(len(self.mesh.vectors)):
            triangle = self.mesh.vectors[i]
            x = triangle[:, 0]
            y = triangle[:, 1]
            z = triangle[:, 2]
            
            # Plot the triangle
            self.ax.plot_trisurf(x, y, z, color='red', alpha=0.7)
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Interactive Blood Vessel STL Model')
        
        # Add a toolbar
        plt.tight_layout()
        
        # Show the plot
        plt.show()
    
    def save_screenshot(self, output_file):
        """
        Save a screenshot of the current view.
        
        Args:
            output_file: Path to save the screenshot.
        """
        if self.fig is None:
            print("Error: No interactive session started.")
            return
        
        # Save the figure
        self.fig.savefig(output_file, dpi=300)
        print(f"Screenshot saved to {output_file}")
