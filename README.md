# Blood Vessel STL Analysis System

A Python-based system for extracting geometric information from STL blood vessel models and storing the data in MySQL.

## Overview

This system provides functionality to:
- Read STL format blood vessel geometry files
- Visualize STL blood vessel models with interactive features
- Extract geometric information (vessel segment length, cross-sectional area)
- Analyze topological relationships from STL models
- Store vessel geometry and topology information in MySQL

## Project Structure

The project follows the Z154 folder structure:

```
blood_vessel_analysis/
├── examples/              # Example scripts
├── src/                   # Source code
│   ├── stl_processing/    # STL file handling
│   ├── visualization/     # 3D visualization
│   ├── geometric_analysis/# Geometric measurements
│   ├── topology/          # Vessel network topology
│   ├── database/          # Database integration
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
└── docs/                  # Documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blood-vessel-analysis.git
cd blood-vessel-analysis

# Install dependencies
pip install -r requirements.txt

# Set up the database configuration
cp .env.example .env
# Edit .env with your MySQL database credentials
```

## Usage

### Analyzing an STL file

```bash
python examples/analyze_stl.py path/to/your/vessel.stl
```

### Visualizing an STL file

```bash
python examples/visualize_stl.py path/to/your/vessel.stl
```

### Using the database functionality

```bash
# Store an STL file in the database
python examples/database_example.py --stl-file path/to/your/vessel.stl

# Retrieve a model from the database
python examples/database_example.py --model-id model_id

# List all models in the database
python examples/database_example.py --list-models

# Delete a model from the database
python examples/database_example.py --delete-model model_id

# Search for models in the database
python examples/database_example.py --search "search_term"

# Show statistics about the models in the database
python examples/database_example.py --statistics
```

## Features

### STL Processing
- Read and validate STL files
- Extract mesh information (vertices, faces, volume, surface area)
- Repair common mesh issues

### Geometric Analysis
- Extract centerlines from blood vessel models
- Calculate vessel segment lengths and diameters
- Compute cross-sectional areas at different points

### Topology Analysis
- Identify branch points and endpoints
- Extract vessel network topology
- Calculate bifurcation angles

### Visualization
- 3D visualization of blood vessel models
- Interactive viewing and manipulation
- Cross-section visualization

### Database Integration
- Store vessel models in MySQL database
- Query and retrieve models based on various criteria
- Calculate statistics across multiple models

## Dependencies

- numpy-stl: STL file processing
- trimesh: 3D mesh manipulation
- pyvista: 3D visualization
- vedo: Advanced 3D visualization
- scipy: Scientific computing
- networkx: Graph-based topology analysis
- mysql-connector-python: MySQL database connectivity
- sqlalchemy: SQL toolkit and ORM

## License

[MIT License](LICENSE)
