# Blood Vessel STL Analysis System - User Guide

## Introduction

The Blood Vessel STL Analysis System is a Python-based tool designed to extract geometric information from STL blood vessel models and store the data in MySQL. This guide will help you understand how to install, configure, and use the system effectively.

## Installation

### Prerequisites

- Python 3.8 or higher
- MySQL 5.7 or higher
- Git

### Setup Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/blood-vessel-analysis.git
   cd blood-vessel-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure the database connection:
   ```bash
   cp .env.example .env
   ```
   
4. Edit the `.env` file with your MySQL database credentials:
   ```
   MYSQL_HOST=localhost
   MYSQL_USER=your_username
   MYSQL_PASSWORD=your_password
   MYSQL_DATABASE=blood_vessel_db
   MYSQL_PORT=3306
   ```

## Running the System

The system provides several example scripts to demonstrate its functionality:

### Analyzing an STL File

To analyze a blood vessel STL model and extract geometric information:

```bash
python examples/analyze_stl.py path/to/your/vessel.stl
```

Optional arguments:
- `--output-dir`, `-o`: Directory to save the output files (default: `output`)
- `--method`, `-m`: Method to use for centerline extraction (`skeleton` or `medial_axis`, default: `skeleton`)

Example:
```bash
python examples/analyze_stl.py data/sample_vessel.stl --output-dir results --method skeleton
```

### Visualizing an STL File

To visualize a blood vessel STL model in 3D:

```bash
python examples/visualize_stl.py path/to/your/vessel.stl
```

Optional arguments:
- `--show-centerline`, `-c`: Show the extracted centerline (default: `False`)
- `--show-cross-sections`, `-s`: Show cross-sections along the centerline (default: `False`)
- `--num-sections`, `-n`: Number of cross-sections to show (default: `10`)

Example:
```bash
python examples/visualize_stl.py data/sample_vessel.stl --show-centerline --show-cross-sections --num-sections 15
```

### Using the Database Functionality

To interact with the MySQL database:

```bash
python examples/database_example.py [options]
```

Available options:
- `--stl-file`: Path to the STL file to analyze and store
- `--model-id`: ID of the model to retrieve from the database
- `--list-models`: List all models in the database
- `--delete-model`: ID of the model to delete from the database
- `--search`: Search term for models in the database
- `--statistics`: Show statistics about the models in the database
- `--host`: MySQL host (default: `localhost`)
- `--user`: MySQL user (default: `root`)
- `--password`: MySQL password (default: `''`)
- `--database`: MySQL database name (default: `blood_vessel_db`)
- `--port`: MySQL port (default: `3306`)

Examples:
```bash
# Store an STL file in the database
python examples/database_example.py --stl-file data/sample_vessel.stl

# Retrieve a model from the database
python examples/database_example.py --model-id sample_vessel

# List all models in the database
python examples/database_example.py --list-models

# Delete a model from the database
python examples/database_example.py --delete-model sample_vessel

# Search for models in the database
python examples/database_example.py --search "sample"

# Show statistics about the models in the database
python examples/database_example.py --statistics
```

## Programmatic Usage

You can also use the system programmatically in your own Python scripts:

```python
from src.core import BloodVesselAnalyzer
from src.database.vessel_repository import VesselRepository

# Create a blood vessel analyzer
analyzer = BloodVesselAnalyzer()

# Load an STL file
analyzer.load_stl("path/to/your/vessel.stl")

# Extract the centerline
centerline = analyzer.extract_centerline()

# Analyze the geometry
results = analyzer.analyze_geometry()

# Build the vessel model
model = analyzer.build_vessel_model()

# Save the model to a JSON file
analyzer.save_model_to_json("vessel_model.json")

# Store the model in the database
repository = VesselRepository()
repository.save_model(model)
```
