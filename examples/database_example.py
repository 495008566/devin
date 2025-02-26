"""
Example script for using the database functionality.

This script demonstrates how to use the blood vessel analysis system to store
and retrieve blood vessel data in a MySQL database.
"""

import os
import sys
import argparse
import json

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core import BloodVesselAnalyzer
from src.database.mysql_connector import MySQLConnector
from src.database.vessel_repository import VesselRepository


def main():
    """Main function for the example script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Demonstrate database functionality.')
    parser.add_argument('--stl-file', help='Path to the STL file to analyze and store')
    parser.add_argument('--model-id', help='ID of the model to retrieve from the database')
    parser.add_argument('--list-models', action='store_true', help='List all models in the database')
    parser.add_argument('--delete-model', help='ID of the model to delete from the database')
    parser.add_argument('--search', help='Search term for models in the database')
    parser.add_argument('--statistics', action='store_true', help='Show statistics about the models in the database')
    parser.add_argument('--host', default='localhost', help='MySQL host')
    parser.add_argument('--user', default='root', help='MySQL user')
    parser.add_argument('--password', default='', help='MySQL password')
    parser.add_argument('--database', default='blood_vessel_db', help='MySQL database name')
    parser.add_argument('--port', type=int, default=3306, help='MySQL port')
    args = parser.parse_args()

    # Create a MySQL connector with the provided configuration
    connector = MySQLConnector({
        "host": args.host,
        "user": args.user,
        "password": args.password,
        "database": args.database,
        "port": args.port
    })

    # Create a vessel repository
    repository = VesselRepository(connector)

    # Connect to the database
    print(f"Connecting to MySQL database at {args.host}:{args.port}...")
    if not connector.connect():
        print("Error: Failed to connect to the database.")
        return 1

    # Create the database and tables if they don't exist
    print("Creating database and tables if they don't exist...")
    connector.create_database()
    if not connector.create_tables():
        print("Error: Failed to create database tables.")
        connector.disconnect()
        return 1

    # Process the command-line arguments
    if args.stl_file:
        # Analyze and store the STL file
        if not os.path.exists(args.stl_file):
            print(f"Error: STL file '{args.stl_file}' does not exist.")
            repository.disconnect()
            return 1

        # Create a blood vessel analyzer
        analyzer = BloodVesselAnalyzer()

        # Load the STL file
        print(f"Loading STL file: {args.stl_file}")
        success = analyzer.load_stl(args.stl_file)
        if not success:
            print("Error: Failed to load the STL file.")
            repository.disconnect()
            return 1

        # Extract the centerline
        print("Extracting centerline...")
        centerline = analyzer.extract_centerline()
        print(f"Centerline extracted with {len(centerline)} points.")

        # Analyze the geometry
        print("Analyzing geometry...")
        results = analyzer.analyze_geometry()

        # Build the vessel model
        print("Building vessel model...")
        model = analyzer.build_vessel_model()

        # Save the model to the database
        print("Saving model to the database...")
        if repository.save_model(model):
            print(f"Model '{model.model_id}' saved to the database.")
        else:
            print("Error: Failed to save the model to the database.")
            repository.disconnect()
            return 1

    elif args.model_id:
        # Retrieve the model from the database
        print(f"Retrieving model '{args.model_id}' from the database...")
        model = repository.get_model(args.model_id)

        if model:
            print(f"Model '{model.model_id}' retrieved from the database.")
            print(f"Name: {model.name}")
            print(f"Filename: {model.get_property('filename', '')}")
            print(f"Surface Area: {model.get_property('surface_area', 0):.2f} square units")
            print(f"Volume: {model.get_property('mesh_volume', 0):.2f} cubic units")
            print(f"Centerline Length: {model.calculate_total_length():.2f} units")
            print(f"Number of Branch Points: {len(model.get_branch_points())}")
            print(f"Number of Endpoints: {len(model.get_endpoints())}")
            print(f"Number of Segments: {len(model.get_all_segments())}")

            # Save the model to a JSON file
            output_filename = f"{model.model_id}_from_db.json"
            print(f"Saving model to {output_filename}...")
            model.save_to_json(output_filename)
            print(f"Model saved to {output_filename}.")
        else:
            print(f"Error: Model '{args.model_id}' not found in the database.")

    elif args.list_models:
        # List all models in the database
        print("Listing all models in the database...")
        models = repository.get_all_models()

        if models:
            print(f"Found {len(models)} models:")
            for model in models:
                print(f"  {model.model_id} - {model.name}")
                print(f"    Surface Area: {model.get_property('surface_area', 0):.2f} square units")
                print(f"    Volume: {model.get_property('volume', 0):.2f} cubic units")
                print(f"    Centerline Length: {model.calculate_total_length():.2f} units")
                print(f"    Branch Points: {len(model.get_branch_points())}")
                print(f"    Endpoints: {len(model.get_endpoints())}")
                print(f"    Segments: {len(model.get_all_segments())}")
                print()
        else:
            print("No models found in the database.")

    elif args.delete_model:
        # Delete the model from the database
        print(f"Deleting model '{args.delete_model}' from the database...")
        if repository.delete_model(args.delete_model):
            print(f"Model '{args.delete_model}' deleted from the database.")
        else:
            print(f"Error: Failed to delete model '{args.delete_model}' from the database.")

    elif args.search:
        # Search for models in the database
        print(f"Searching for models matching '{args.search}'...")
        # Create a search criteria dictionary
        criteria = {"name": args.search}
        models = repository.search_models(criteria)

        if models:
            print(f"Found {len(models)} matching models:")
            for model in models:
                print(f"  {model.model_id} - {model.name}")
                print(f"    Surface Area: {model.get_property('surface_area', 0):.2f} square units")
                print(f"    Volume: {model.get_property('volume', 0):.2f} cubic units")
                print(f"    Centerline Length: {model.calculate_total_length():.2f} units")
                print(f"    Branch Points: {len(model.get_branch_points())}")
                print(f"    Endpoints: {len(model.get_endpoints())}")
                print(f"    Segments: {len(model.get_all_segments())}")
                print()
        else:
            print(f"No models found matching '{args.search}'.")

    elif args.statistics:
        # Show statistics about the models in the database
        print("Getting statistics about the models in the database...")
        # Get all models
        models = repository.get_all_models()
        
        if models:
            # Calculate statistics
            num_models = len(models)
            total_surface_area = sum(model.get_property('surface_area', 0) for model in models)
            total_volume = sum(model.get_property('volume', 0) for model in models)
            total_centerline_length = sum(model.calculate_total_length() for model in models)
            total_branch_points = sum(len(model.get_branch_points()) for model in models)
            total_endpoints = sum(len(model.get_endpoints()) for model in models)
            total_segments = sum(len(model.get_all_segments()) for model in models)
            
            # Calculate averages
            avg_surface_area = total_surface_area / num_models if num_models > 0 else 0
            avg_volume = total_volume / num_models if num_models > 0 else 0
            avg_centerline_length = total_centerline_length / num_models if num_models > 0 else 0
            avg_branch_points = total_branch_points / num_models if num_models > 0 else 0
            avg_endpoints = total_endpoints / num_models if num_models > 0 else 0
            avg_segments = total_segments / num_models if num_models > 0 else 0
            
            print(f"Number of Models: {num_models}")
            print(f"Average Surface Area: {avg_surface_area:.2f} square units")
            print(f"Average Volume: {avg_volume:.2f} cubic units")
            print(f"Average Centerline Length: {avg_centerline_length:.2f} units")
            print(f"Average Number of Branch Points: {avg_branch_points:.2f}")
            print(f"Average Number of Endpoints: {avg_endpoints:.2f}")
            print(f"Average Number of Segments: {avg_segments:.2f}")
        else:
            print("No models found in the database.")

    else:
        # No command-line arguments provided
        print("No action specified. Use --help to see available options.")

    # Disconnect from the database
    connector.disconnect()
    print("Disconnected from the database.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
