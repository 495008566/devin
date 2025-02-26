"""
Example script for visualizing a blood vessel STL model.

This script demonstrates how to use the blood vessel analysis system to visualize
STL blood vessel models with interactive features.
"""

import os
import sys
import argparse
import numpy as np

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import BloodVesselAnalyzer
from src.visualization.stl_visualizer import STLVisualizer
from src.visualization.interactive_viewer import InteractiveViewer


def main():
    """Main function for the example script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize a blood vessel STL model.')
    parser.add_argument('stl_file', help='Path to the STL file')
    parser.add_argument('--interactive', '-i', action='store_true', help='Enable interactive mode')
    parser.add_argument('--output', '-o', help='Path to save the visualization image')
    parser.add_argument('--animation', '-a', help='Path to save the animation')
    parser.add_argument('--method', '-m', default='skeleton', choices=['skeleton', 'medial_axis'],
                        help='Method to use for centerline extraction')
    args = parser.parse_args()

    # Check if the STL file exists
    if not os.path.exists(args.stl_file):
        print(f"Error: STL file '{args.stl_file}' does not exist.")
        return 1

    # Create a blood vessel analyzer
    analyzer = BloodVesselAnalyzer()

    # Load the STL file
    print(f"Loading STL file: {args.stl_file}")
    success = analyzer.load_stl(args.stl_file)
    if not success:
        print("Error: Failed to load the STL file.")
        return 1

    # Extract the centerline
    print(f"Extracting centerline using method: {args.method}")
    centerline = analyzer.extract_centerline(method=args.method)
    print(f"Centerline extracted with {len(centerline)} points.")

    # Analyze the geometry
    print("Analyzing geometry...")
    results = analyzer.analyze_geometry()

    # Print a summary of the results
    print("\nAnalysis Results:")
    print(f"Surface Area: {results.get('surface_area', 0):.2f} square units")
    print(f"Centerline Length: {results.get('centerline_length', 0):.2f} units")
    print(f"Number of Branch Points: {results.get('num_branch_points', 0)}")
    print(f"Number of Endpoints: {results.get('num_endpoints', 0)}")
    print(f"Number of Segments: {results.get('num_segments', 0)}")

    # Get the data for visualization
    mesh = analyzer.stl_reader.mesh
    centerline = analyzer.vessel_model.get_property("centerline")
    branch_points = analyzer.vessel_model.get_property("branch_points")
    endpoints = analyzer.vessel_model.get_property("endpoints")
    segments = analyzer.vessel_model.get_property("segments")
    cross_sections = analyzer.vessel_model.get_property("cross_sections")

    if args.interactive:
        # Create an interactive viewer
        print("Creating interactive viewer...")
        viewer = InteractiveViewer()
        viewer.set_mesh(mesh)
        viewer.set_centerline(centerline)
        viewer.set_branch_points(branch_points)
        viewer.set_endpoints(endpoints)
        viewer.set_segments(segments)
        viewer.set_cross_sections(cross_sections)

        # Add some predefined views
        if branch_points is not None and len(branch_points) > 0:
            # Add a view centered on each branch point
            for i, point in enumerate(branch_points):
                viewer.add_view(f"Branch Point {i+1}", camera_position=point + np.array([10, 10, 10]), focal_point=point)

        if endpoints is not None and len(endpoints) > 0:
            # Add a view centered on each endpoint
            for i, point in enumerate(endpoints):
                viewer.add_view(f"Endpoint {i+1}", camera_position=point + np.array([10, 10, 10]), focal_point=point)

        # Run the interactive viewer
        print("Running interactive viewer...")
        viewer.run(interactive=True)
    else:
        # Create a non-interactive visualization
        print("Creating visualization...")
        visualizer = STLVisualizer()
        visualizer.set_mesh(mesh)
        visualizer.set_centerline(centerline)
        visualizer.set_branch_points(branch_points)
        visualizer.set_endpoints(endpoints)
        visualizer.set_segments(segments)
        visualizer.set_cross_sections(cross_sections)

        # Create a plotter
        visualizer.create_plotter(off_screen=args.output is not None)

        # Visualize everything
        visualizer.visualize_all()

        # Add axes
        visualizer.add_axes()

        # Create an animation if requested
        if args.animation:
            print(f"Creating animation: {args.animation}")
            visualizer.create_animation(args.animation)

        # Save the visualization if requested
        if args.output:
            print(f"Saving visualization to: {args.output}")
            visualizer.show(interactive=False, screenshot=args.output)
        else:
            # Show the visualization
            print("Showing visualization...")
            visualizer.show(interactive=True)

    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
