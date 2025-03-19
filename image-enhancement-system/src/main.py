#!/usr/bin/env python3
"""
Main entry point for the Image Enhancement System.
"""

import os
import sys

# Add the parent directory to the path to make imports work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import modules directly
from core import processor
from ui import interface
from database import db_manager
from utils import helpers

def main():
    """Main function to run the Image Enhancement System."""
    print("Starting Image Enhancement System...")
    
    # Initialize components
    db = db_manager.DatabaseManager()
    app = interface.Application(db)
    
    # Run the application
    app.run()

if __name__ == "__main__":
    main()
