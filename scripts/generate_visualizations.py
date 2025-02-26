"""
Script to generate visualizations of blood vessel STL models.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stl_processing.stl_reader import STLReader
from src.geometric_analysis.centerline import CenterlineExtractor
from src.geometric_analysis.cross_section import CrossSectionAnalyzer
from src.topology.vessel_network import VesselNetwork
from src.visualization.stl_visualizer import STLVisualizer
from src.core import BloodVesselAnalyzer

# Create output directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

def generate_basic_visualization(stl_file, output_file):
    """
    Generate a basic visualization of an STL file.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Create a visualizer
    visualizer = STLVisualizer()
    visualizer.mesh = analyzer.reader.mesh
    
    # Generate the visualization
    print(f"Generating visualization: {output_file}")
    visualizer.save_visualization(output_file)
    
    print(f"Visualization saved to {output_file}")

def generate_centerline_visualization(stl_file, output_file):
    """
    Generate a visualization of an STL file with its centerline.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Extract the centerline
    print("Extracting centerline...")
    centerline = analyzer.extract_centerline()
    
    # Create a visualizer
    visualizer = STLVisualizer()
    visualizer.mesh = analyzer.reader.mesh
    
    # Generate the visualization with centerline
    print(f"Generating visualization with centerline: {output_file}")
    visualizer.save_visualization_with_centerline(output_file, centerline)
    
    print(f"Visualization saved to {output_file}")

def generate_cross_section_visualization(stl_file, output_file, num_sections=5):
    """
    Generate a visualization of cross-sections along the centerline.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
        num_sections: Number of cross-sections to visualize.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Extract the centerline
    print("Extracting centerline...")
    centerline = analyzer.extract_centerline()
    
    # Analyze cross-sections
    print(f"Analyzing {num_sections} cross-sections...")
    cross_sections = analyzer.analyze_cross_sections(centerline, num_sections)
    
    # Create a visualizer
    visualizer = STLVisualizer()
    visualizer.mesh = analyzer.reader.mesh
    
    # Generate the visualization with cross-sections
    print(f"Generating visualization with cross-sections: {output_file}")
    visualizer.save_visualization_with_cross_sections(output_file, centerline, cross_sections)
    
    print(f"Visualization saved to {output_file}")

def generate_topology_visualization(stl_file, output_file):
    """
    Generate a visualization of the vessel topology.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Extract the centerline
    print("Extracting centerline...")
    centerline = analyzer.extract_centerline()
    
    # Build the vessel model
    print("Building vessel model...")
    model = analyzer.build_vessel_model()
    
    # Create a visualizer
    visualizer = STLVisualizer()
    visualizer.mesh = analyzer.reader.mesh
    
    # Generate the visualization with topology
    print(f"Generating visualization with topology: {output_file}")
    visualizer.save_visualization_with_topology(output_file, model)
    
    print(f"Visualization saved to {output_file}")

def generate_geometric_analysis_visualization(stl_file, output_file):
    """
    Generate a visualization of the geometric analysis results.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Analyze the geometry
    print("Analyzing geometry...")
    results = analyzer.analyze_geometry()
    
    # Create a figure for the results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the results
    labels = list(results.keys())
    values = list(results.values())
    
    # Filter out non-numeric values
    numeric_labels = []
    numeric_values = []
    for label, value in zip(labels, values):
        if isinstance(value, (int, float)):
            numeric_labels.append(label)
            numeric_values.append(value)
    
    ax.bar(numeric_labels, numeric_values)
    ax.set_title(f"Geometric Analysis Results for {os.path.basename(stl_file)}")
    ax.set_ylabel("Value")
    ax.set_xlabel("Metric")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    # Save the figure
    print(f"Saving geometric analysis visualization: {output_file}")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def generate_cross_section_area_plot(stl_file, output_file, num_sections=20):
    """
    Generate a plot of cross-sectional areas along the centerline.
    
    Args:
        stl_file: Path to the STL file.
        output_file: Path to save the visualization.
        num_sections: Number of cross-sections to analyze.
    """
    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()
    
    # Load the STL file
    print(f"Loading STL file: {stl_file}")
    success = analyzer.load_stl(stl_file)
    if not success:
        print(f"Error: Failed to load STL file {stl_file}")
        return
    
    # Extract the centerline
    print("Extracting centerline...")
    centerline = analyzer.extract_centerline()
    
    # Analyze cross-sections
    print(f"Analyzing {num_sections} cross-sections...")
    cross_sections = analyzer.analyze_cross_sections(centerline, num_sections)
    
    # Extract cross-section areas and positions
    positions = np.linspace(0, 1, len(cross_sections))
    areas = [cs.get("area", 0) for cs in cross_sections]
    
    # Create a figure for the results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the results
    ax.plot(positions, areas, 'b-', linewidth=2)
    ax.set_title(f"Cross-Sectional Area Along Centerline for {os.path.basename(stl_file)}")
    ax.set_ylabel("Cross-Sectional Area")
    ax.set_xlabel("Position Along Centerline (Normalized)")
    ax.grid(True)
    plt.tight_layout()
    
    # Save the figure
    print(f"Saving cross-section area plot: {output_file}")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")

def generate_all_visualizations():
    """
    Generate all visualizations for all sample STL files.
    """
    # Get all STL files in the sample_stl directory
    stl_files = [f for f in os.listdir("sample_stl") if f.endswith(".stl")]
    
    for stl_file in stl_files:
        stl_path = os.path.join("sample_stl", stl_file)
        base_name = os.path.splitext(stl_file)[0]
        
        # Generate basic visualization
        output_file = os.path.join("visualizations", f"{base_name}_basic.png")
        generate_basic_visualization(stl_path, output_file)
        
        # Generate centerline visualization
        output_file = os.path.join("visualizations", f"{base_name}_centerline.png")
        generate_centerline_visualization(stl_path, output_file)
        
        # Generate cross-section visualization
        output_file = os.path.join("visualizations", f"{base_name}_cross_sections.png")
        generate_cross_section_visualization(stl_path, output_file)
        
        # Generate topology visualization
        output_file = os.path.join("visualizations", f"{base_name}_topology.png")
        generate_topology_visualization(stl_path, output_file)
        
        # Generate geometric analysis visualization
        output_file = os.path.join("visualizations", f"{base_name}_geometry.png")
        generate_geometric_analysis_visualization(stl_path, output_file)
        
        # Generate cross-section area plot
        output_file = os.path.join("visualizations", f"{base_name}_cross_section_areas.png")
        generate_cross_section_area_plot(stl_path, output_file)

if __name__ == "__main__":
    generate_all_visualizations()
