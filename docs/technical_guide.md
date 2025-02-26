# Blood Vessel STL Analysis System - Technical Guide

## System Architecture

The Blood Vessel STL Analysis System follows a modular architecture with the following components:

1. **STL Processing**: Handles reading and validating STL files
2. **Geometric Analysis**: Extracts centerlines and computes geometric properties
3. **Topology Analysis**: Analyzes the vessel network topology
4. **Visualization**: Provides 3D visualization capabilities
5. **Database Integration**: Stores and retrieves vessel models from MySQL
6. **Core**: Coordinates the different components

## Z154 Folder Structure

The project follows the Z154 folder structure, which organizes the code into logical modules:

```
blood_vessel_analysis/
├── examples/              # Example scripts
│   ├── analyze_stl.py     # Example for analyzing STL files
│   ├── visualize_stl.py   # Example for visualizing STL files
│   └── database_example.py# Example for database operations
├── src/                   # Source code
│   ├── stl_processing/    # STL file handling
│   │   ├── __init__.py
│   │   └── stl_reader.py  # STL file reader
│   ├── visualization/     # 3D visualization
│   │   ├── __init__.py
│   │   ├── stl_visualizer.py      # STL visualization
│   │   └── interactive_viewer.py  # Interactive viewer
│   ├── geometric_analysis/# Geometric measurements
│   │   ├── __init__.py
│   │   ├── centerline.py  # Centerline extraction
│   │   └── cross_section.py # Cross-section analysis
│   ├── topology/          # Vessel network topology
│   │   ├── __init__.py
│   │   └── vessel_network.py # Vessel network analysis
│   ├── database/          # Database integration
│   │   ├── __init__.py
│   │   ├── mysql_connector.py # MySQL connection
│   │   └── vessel_repository.py # Vessel data repository
│   ├── utils/             # Utility functions
│   │   ├── __init__.py
│   │   ├── geometry.py    # Geometry utilities
│   │   └── data_structures.py # Data structures
│   ├── __init__.py
│   └── core.py            # Core functionality
├── tests/                 # Unit and integration tests
│   ├── test_stl_reader.py
│   ├── test_centerline.py
│   ├── test_cross_section.py
│   ├── test_vessel_network.py
│   ├── test_data_structures.py
│   └── test_integration.py
└── docs/                  # Documentation
    ├── user_guide.md      # User guide
    └── technical_guide.md # Technical guide
```

## Component Details

### STL Processing

The STL processing module (`src/stl_processing/`) handles reading and validating STL files. It uses the `numpy-stl` and `trimesh` libraries to load and process STL files.

Key classes:
- `STLReader`: Reads STL files and provides mesh information

Key functionality:
- Reading STL files
- Extracting mesh information (vertices, faces, volume, surface area)
- Validating mesh integrity
- Repairing common mesh issues

### Geometric Analysis

The geometric analysis module (`src/geometric_analysis/`) extracts centerlines and computes geometric properties of blood vessels.

Key classes:
- `CenterlineExtractor`: Extracts centerlines from blood vessel models
- `CrossSectionAnalyzer`: Computes cross-sections along centerlines

Key functionality:
- Centerline extraction using skeletonization or medial axis transform
- Branch point and endpoint identification
- Segment length and diameter calculation
- Cross-sectional area computation

### Topology Analysis

The topology analysis module (`src/topology/`) analyzes the vessel network topology.

Key classes:
- `VesselNetwork`: Represents the vessel network topology

Key functionality:
- Bifurcation angle calculation
- Vessel segment connectivity analysis
- Network graph representation

### Visualization

The visualization module (`src/visualization/`) provides 3D visualization capabilities.

Key classes:
- `STLVisualizer`: Visualizes STL models
- `InteractiveViewer`: Provides interactive viewing capabilities

Key functionality:
- 3D visualization of blood vessel models
- Centerline visualization
- Cross-section visualization
- Interactive manipulation (rotation, zoom, pan)

### Database Integration

The database integration module (`src/database/`) handles storing and retrieving vessel models from MySQL.

Key classes:
- `MySQLConnector`: Connects to MySQL database
- `VesselRepository`: Stores and retrieves vessel models

Key functionality:
- Database connection management
- Table creation and schema management
- Model storage and retrieval
- Model search and statistics

### Core

The core module (`src/core.py`) coordinates the different components.

Key classes:
- `BloodVesselAnalyzer`: Main class for analyzing blood vessel models

Key functionality:
- STL file loading
- Centerline extraction
- Geometric analysis
- Vessel model building
- Result export

## Data Structures

The system uses several data structures to represent vessel geometry and topology:

- `VesselModel`: Represents a complete vessel model
- `VesselSegment`: Represents a vessel segment
- `VesselNode`: Represents a vessel node (branch point or endpoint)
- `VesselDatabase`: Manages a collection of vessel models

These data structures are defined in `src/utils/data_structures.py`.

## Database Schema

The system uses the following database schema:

### vessel_models

Stores basic information about vessel models.

| Column | Type | Description |
|--------|------|-------------|
| model_id | VARCHAR(255) | Primary key |
| name | VARCHAR(255) | Model name |
| created_at | TIMESTAMP | Creation timestamp |
| updated_at | TIMESTAMP | Update timestamp |
| surface_area | FLOAT | Surface area |
| volume | FLOAT | Volume |
| centerline_length | FLOAT | Centerline length |
| num_branch_points | INT | Number of branch points |
| num_endpoints | INT | Number of endpoints |
| num_segments | INT | Number of segments |
| metadata | JSON | Additional metadata |

### vessel_segments

Stores information about vessel segments.

| Column | Type | Description |
|--------|------|-------------|
| segment_id | INT | Segment ID |
| model_id | VARCHAR(255) | Model ID (foreign key) |
| start_node_id | VARCHAR(255) | Start node ID |
| end_node_id | VARCHAR(255) | End node ID |
| length | FLOAT | Segment length |
| avg_diameter | FLOAT | Average diameter |
| min_diameter | FLOAT | Minimum diameter |
| max_diameter | FLOAT | Maximum diameter |
| avg_cross_section_area | FLOAT | Average cross-section area |
| properties | JSON | Additional properties |

### vessel_nodes

Stores information about vessel nodes.

| Column | Type | Description |
|--------|------|-------------|
| node_id | VARCHAR(255) | Node ID |
| model_id | VARCHAR(255) | Model ID (foreign key) |
| node_type | VARCHAR(50) | Node type (branch, endpoint) |
| x | FLOAT | X coordinate |
| y | FLOAT | Y coordinate |
| z | FLOAT | Z coordinate |
| properties | JSON | Additional properties |

### vessel_cross_sections

Stores information about vessel cross-sections.

| Column | Type | Description |
|--------|------|-------------|
| cross_section_id | INT | Primary key |
| segment_id | INT | Segment ID |
| model_id | VARCHAR(255) | Model ID |
| position | FLOAT | Position along segment |
| area | FLOAT | Cross-section area |
| perimeter | FLOAT | Cross-section perimeter |
| max_diameter | FLOAT | Maximum diameter |
| min_diameter | FLOAT | Minimum diameter |
| circularity | FLOAT | Circularity |
| properties | JSON | Additional properties |

## Testing

The system includes unit and integration tests in the `tests/` directory. To run the tests:

```bash
cd tests
python -m unittest discover
```

## Dependencies

The system relies on the following key dependencies:

- `numpy-stl`: STL file processing
- `trimesh`: 3D mesh manipulation
- `pyvista`: 3D visualization
- `vedo`: Advanced 3D visualization
- `scipy`: Scientific computing
- `networkx`: Graph-based topology analysis
- `mysql-connector-python`: MySQL database connectivity
- `sqlalchemy`: SQL toolkit and ORM
