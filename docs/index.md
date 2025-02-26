# Blood Vessel STL Analysis System Documentation

Welcome to the documentation for the Blood Vessel STL Analysis System. This system provides functionality for extracting geometric information from STL blood vessel models and storing the data in MySQL.

## Documentation Contents

- [User Guide](user_guide.md): Instructions for installing, configuring, and using the system
- [Technical Guide](technical_guide.md): Details about the system architecture, folder structure, and components
- [API Reference](api_reference.md): Documentation of all classes and methods in the system

## Overview

The Blood Vessel STL Analysis System is a Python-based tool designed to:

- Read STL format blood vessel geometry files
- Visualize STL blood vessel models with interactive features
- Extract geometric information (vessel segment length, cross-sectional area)
- Analyze topological relationships from STL models
- Store vessel geometry and topology information in MySQL

## Quick Start

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
   # Edit .env with your MySQL database credentials
   ```

4. Analyze an STL file:
   ```bash
   python examples/analyze_stl.py path/to/your/vessel.stl
   ```

5. Visualize an STL file:
   ```bash
   python examples/visualize_stl.py path/to/your/vessel.stl
   ```

6. Use the database functionality:
   ```bash
   python examples/database_example.py --stl-file path/to/your/vessel.stl
   ```

For more detailed information, please refer to the [User Guide](user_guide.md).
