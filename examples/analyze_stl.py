"""
Example script for analyzing a blood vessel STL model.

This script demonstrates how to use the blood vessel analysis system to extract
geometric information from an STL blood vessel model.
"""

import os
import sys
import argparse
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import BloodVesselAnalyzer


def main():
    """Main function for the example script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Analyze a blood vessel STL model.')
    parser.add_argument('stl_file', help='Path to the STL file')
    parser.add_argument('--output-dir', '-o', default='output', help='Directory to save the output files')
    parser.add_argument('--method', '-m', default='skeleton', choices=['skeleton', 'medial_axis'],
                        help='Method to use for centerline extraction')
    args = parser.parse_args()

    # Check if the STL file exists
    if not os.path.exists(args.stl_file):
        print(f"Error: STL file '{args.stl_file}' does not exist.")
        return 1

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

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

    # Build the vessel model
    print("Building vessel model...")
    model = analyzer.build_vessel_model()

    # Export the results
    print(f"Exporting results to: {args.output_dir}")
    analyzer.export_results(args.output_dir)

    # Save the analysis results to a JSON file
    results_filename = os.path.join(args.output_dir, "analysis_results.json")
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))

    print(f"Analysis results saved to: {results_filename}")
    print("Done!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
