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
├── data/                  # Sample datasets and test files
├── docs/                  # Documentation
├── src/                   # Source code
│   ├── stl_processing/    # STL file handling
│   ├── visualization/     # 3D visualization
│   ├── geometric_analysis/# Geometric measurements
│   ├── topology/          # Vessel network topology
│   ├── database/          # Database integration
│   └── utils/             # Utility functions
├── tests/                 # Unit and integration tests
├── scripts/               # Utility scripts
└── examples/              # Example usage and tutorials
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/blood_vessel_analysis.git
cd blood_vessel_analysis

# Install dependencies
pip install -r requirements.txt
```

## Usage

Documentation and examples will be added as the project develops.

## License

[MIT License](LICENSE)
