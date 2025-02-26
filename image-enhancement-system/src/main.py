#!/usr/bin/env python3
"""
Main entry point for the Image Enhancement System.
"""

import os
import sys

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import processor
from src.ui import interface
from src.database import db_manager
from src.utils import helpers

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
