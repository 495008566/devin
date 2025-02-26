# Blood Vessel STL Analysis System - API Reference

## Core Module

### BloodVesselAnalyzer

The main class for analyzing blood vessel models.

```python
from src.core import BloodVesselAnalyzer
```

#### Methods

##### `__init__()`

Initialize the blood vessel analyzer.

##### `load_stl(filename: str) -> bool`

Load an STL file.

- **Parameters**:
  - `filename`: Path to the STL file
- **Returns**: `True` if the file was loaded successfully, `False` otherwise

##### `extract_centerline(method: str = 'skeleton') -> np.ndarray`

Extract the centerline from the loaded STL file.

- **Parameters**:
  - `method`: Method to use for centerline extraction (`'skeleton'` or `'medial_axis'`)
- **Returns**: Numpy array of centerline points

##### `analyze_cross_sections(centerline: np.ndarray = None, num_sections: int = 10) -> List[Dict[str, Any]]`

Analyze cross-sections along the centerline.

- **Parameters**:
  - `centerline`: Centerline points (if `None`, uses the extracted centerline)
  - `num_sections`: Number of cross-sections to analyze
- **Returns**: List of dictionaries containing cross-section information

##### `analyze_geometry() -> Dict[str, Any]`

Analyze the geometry of the loaded STL file.

- **Returns**: Dictionary containing geometric information

##### `build_vessel_model() -> VesselModel`

Build a vessel model from the analyzed data.

- **Returns**: `VesselModel` object

##### `save_model_to_json(filename: str) -> bool`

Save the vessel model to a JSON file.

- **Parameters**:
  - `filename`: Path to the output JSON file
- **Returns**: `True` if the model was saved successfully, `False` otherwise

##### `load_model_from_json(filename: str) -> VesselModel`

Load a vessel model from a JSON file.

- **Parameters**:
  - `filename`: Path to the input JSON file
- **Returns**: `VesselModel` object

##### `export_results(output_dir: str) -> bool`

Export analysis results to the specified directory.

- **Parameters**:
  - `output_dir`: Path to the output directory
- **Returns**: `True` if the results were exported successfully, `False` otherwise

## STL Processing Module

### STLReader

Class for reading and processing STL files.

```python
from src.stl_processing.stl_reader import STLReader
```

#### Methods

##### `__init__()`

Initialize the STL reader.

##### `read_file(filename: str) -> bool`

Read an STL file and store the mesh.

- **Parameters**:
  - `filename`: Path to the STL file
- **Returns**: `True` if the file was read successfully, `False` otherwise

##### `get_mesh_info() -> Dict[str, Any]`

Get basic information about the loaded mesh.

- **Returns**: Dictionary containing mesh information

##### `get_bounding_box() -> Tuple[np.ndarray, np.ndarray]`

Get the bounding box of the mesh.

- **Returns**: Tuple of minimum and maximum points of the bounding box

##### `get_surface_area() -> float`

Calculate the surface area of the mesh.

- **Returns**: Surface area of the mesh

##### `validate_mesh() -> Dict[str, Any]`

Validate the mesh for common issues.

- **Returns**: Dictionary containing validation results

##### `repair_mesh() -> bool`

Attempt to repair common mesh issues.

- **Returns**: `True` if repairs were made, `False` otherwise

## Geometric Analysis Module

### CenterlineExtractor

Class for extracting centerlines from blood vessel models.

```python
from src.geometric_analysis.centerline import CenterlineExtractor
```

#### Methods

##### `__init__(mesh=None)`

Initialize the centerline extractor.

- **Parameters**:
  - `mesh`: The mesh to extract centerlines from

##### `set_mesh(mesh)`

Set the mesh to extract centerlines from.

- **Parameters**:
  - `mesh`: The mesh to extract centerlines from

##### `extract_centerline(method: str = 'skeleton') -> np.ndarray`

Extract the centerline from the mesh.

- **Parameters**:
  - `method`: The method to use for centerline extraction (`'skeleton'` or `'medial_axis'`)
- **Returns**: Numpy array of centerline points

##### `get_centerline_graph() -> Optional[nx.Graph]`

Get the centerline graph.

- **Returns**: NetworkX graph of the centerline, or `None` if not computed

##### `get_branch_points() -> np.ndarray`

Get the branch points of the centerline.

- **Returns**: Numpy array of branch point coordinates

##### `get_endpoints() -> np.ndarray`

Get the endpoints of the centerline.

- **Returns**: Numpy array of endpoint coordinates

##### `get_centerline_segments() -> List[np.ndarray]`

Get the centerline segments.

- **Returns**: List of numpy arrays, each representing a centerline segment

##### `calculate_segment_lengths() -> List[float]`

Calculate the length of each centerline segment.

- **Returns**: List of segment lengths

##### `calculate_segment_diameters() -> List[float]`

Estimate the diameter of each centerline segment.

- **Returns**: List of segment diameters

### CrossSectionAnalyzer

Class for analyzing cross-sections of blood vessel models.

```python
from src.geometric_analysis.cross_section import CrossSectionAnalyzer
```

#### Methods

##### `__init__(mesh=None)`

Initialize the cross-section analyzer.

- **Parameters**:
  - `mesh`: The mesh to analyze

##### `set_mesh(mesh)`

Set the mesh to analyze.

- **Parameters**:
  - `mesh`: The mesh to analyze

##### `compute_cross_section(point: np.ndarray, normal: np.ndarray) -> Dict[str, Any]`

Compute a cross-section at the specified point with the given normal.

- **Parameters**:
  - `point`: Point on the centerline
  - `normal`: Normal vector (direction of the cross-section)
- **Returns**: Dictionary containing cross-section information

##### `compute_cross_sections_along_centerline(centerline: np.ndarray, num_sections: int = 10) -> List[Dict[str, Any]]`

Compute cross-sections along the centerline.

- **Parameters**:
  - `centerline`: Centerline points
  - `num_sections`: Number of cross-sections to compute
- **Returns**: List of dictionaries containing cross-section information

## Topology Module

### VesselNetwork

Class for analyzing vessel network topology.

```python
from src.topology.vessel_network import VesselNetwork
```

#### Methods

##### `__init__(centerline: np.ndarray = None, branch_points: np.ndarray = None, endpoints: np.ndarray = None)`

Initialize the vessel network.

- **Parameters**:
  - `centerline`: Centerline points
  - `branch_points`: Branch point coordinates
  - `endpoints`: Endpoint coordinates

##### `set_centerline(centerline: np.ndarray)`

Set the centerline.

- **Parameters**:
  - `centerline`: Centerline points

##### `set_branch_points(branch_points: np.ndarray)`

Set the branch points.

- **Parameters**:
  - `branch_points`: Branch point coordinates

##### `set_endpoints(endpoints: np.ndarray)`

Set the endpoints.

- **Parameters**:
  - `endpoints`: Endpoint coordinates

##### `build_network() -> nx.Graph`

Build the vessel network graph.

- **Returns**: NetworkX graph of the vessel network

##### `get_bifurcation_angles() -> List[float]`

Calculate the bifurcation angles.

- **Returns**: List of bifurcation angles in degrees

##### `get_segment_connectivity() -> Dict[int, List[int]]`

Get the connectivity of vessel segments.

- **Returns**: Dictionary mapping segment IDs to lists of connected segment IDs

## Visualization Module

### STLVisualizer

Class for visualizing STL models.

```python
from src.visualization.stl_visualizer import STLVisualizer
```

#### Methods

##### `__init__()`

Initialize the STL visualizer.

##### `load_stl(filename: str) -> bool`

Load an STL file.

- **Parameters**:
  - `filename`: Path to the STL file
- **Returns**: `True` if the file was loaded successfully, `False` otherwise

##### `visualize(show_centerline: bool = False, centerline: np.ndarray = None) -> None`

Visualize the loaded STL file.

- **Parameters**:
  - `show_centerline`: Whether to show the centerline
  - `centerline`: Centerline points (if `None` and `show_centerline` is `True`, extracts the centerline)

##### `visualize_with_cross_sections(centerline: np.ndarray = None, num_sections: int = 10) -> None`

Visualize the loaded STL file with cross-sections.

- **Parameters**:
  - `centerline`: Centerline points (if `None`, extracts the centerline)
  - `num_sections`: Number of cross-sections to show

### InteractiveViewer

Class for interactive viewing of STL models.

```python
from src.visualization.interactive_viewer import InteractiveViewer
```

#### Methods

##### `__init__()`

Initialize the interactive viewer.

##### `load_stl(filename: str) -> bool`

Load an STL file.

- **Parameters**:
  - `filename`: Path to the STL file
- **Returns**: `True` if the file was loaded successfully, `False` otherwise

##### `start_interactive_session() -> None`

Start an interactive viewing session.

## Database Module

### MySQLConnector

Class for connecting to a MySQL database.

```python
from src.database.mysql_connector import MySQLConnector
```

#### Methods

##### `__init__(config: Dict[str, Any] = None)`

Initialize the MySQL connector.

- **Parameters**:
  - `config`: Configuration dictionary with connection parameters (if `None`, uses environment variables)

##### `connect() -> bool`

Connect to the MySQL database.

- **Returns**: `True` if the connection was successful, `False` otherwise

##### `disconnect() -> None`

Disconnect from the MySQL database.

##### `execute_query(query: str, params: Tuple = None) -> bool`

Execute a query that does not return results.

- **Parameters**:
  - `query`: The query to execute
  - `params`: The parameters for the query
- **Returns**: `True` if the query was executed successfully, `False` otherwise

##### `execute_select(query: str, params: Tuple = None) -> List[Dict[str, Any]]`

Execute a select query and return the results.

- **Parameters**:
  - `query`: The query to execute
  - `params`: The parameters for the query
- **Returns**: List of dictionaries containing the query results

##### `create_database() -> bool`

Create the database if it does not exist.

- **Returns**: `True` if the database was created successfully, `False` otherwise

##### `create_tables() -> bool`

Create the necessary tables for the blood vessel database.

- **Returns**: `True` if the tables were created successfully, `False` otherwise

### VesselRepository

Class for storing and retrieving vessel models from a database.

```python
from src.database.vessel_repository import VesselRepository
```

#### Methods

##### `__init__(connector: MySQLConnector = None)`

Initialize the vessel repository.

- **Parameters**:
  - `connector`: The MySQL connector to use (if `None`, creates a new connector)

##### `save_model(model: VesselModel) -> bool`

Save a vessel model to the database.

- **Parameters**:
  - `model`: The vessel model to save
- **Returns**: `True` if the model was saved successfully, `False` otherwise

##### `get_model(model_id: str) -> Optional[VesselModel]`

Get a vessel model from the database.

- **Parameters**:
  - `model_id`: The ID of the model to get
- **Returns**: `VesselModel` object, or `None` if not found

##### `delete_model(model_id: str) -> bool`

Delete a vessel model from the database.

- **Parameters**:
  - `model_id`: The ID of the model to delete
- **Returns**: `True` if the model was deleted successfully, `False` otherwise

##### `get_all_models() -> List[VesselModel]`

Get all vessel models from the database.

- **Returns**: List of `VesselModel` objects

##### `search_models(criteria: Dict[str, Any]) -> List[VesselModel]`

Search for vessel models matching the given criteria.

- **Parameters**:
  - `criteria`: Dictionary of search criteria
- **Returns**: List of matching `VesselModel` objects

## Data Structures

### VesselModel

Class for representing a complete vessel model.

```python
from src.utils.data_structures import VesselModel
```

#### Methods

##### `__init__(model_id: str = "", name: str = "")`

Initialize a vessel model.

- **Parameters**:
  - `model_id`: The ID of the model
  - `name`: The name of the model

##### `set_property(key: str, value: Any) -> None`

Set a property of the model.

- **Parameters**:
  - `key`: The property key
  - `value`: The property value

##### `get_property(key: str, default: Any = None) -> Any`

Get a property of the model.

- **Parameters**:
  - `key`: The property key
  - `default`: The default value to return if the property does not exist
- **Returns**: The property value

##### `add_segment(segment: VesselSegment) -> None`

Add a segment to the model.

- **Parameters**:
  - `segment`: The segment to add

##### `add_node(node: VesselNode) -> None`

Add a node to the model.

- **Parameters**:
  - `node`: The node to add

##### `get_segment(segment_id: int) -> Optional[VesselSegment]`

Get a segment by ID.

- **Parameters**:
  - `segment_id`: The ID of the segment
- **Returns**: The segment, or `None` if not found

##### `get_node(node_id: str) -> Optional[VesselNode]`

Get a node by ID.

- **Parameters**:
  - `node_id`: The ID of the node
- **Returns**: The node, or `None` if not found

##### `get_all_segments() -> Dict[int, VesselSegment]`

Get all segments.

- **Returns**: Dictionary mapping segment IDs to segments

##### `get_all_nodes() -> Dict[str, VesselNode]`

Get all nodes.

- **Returns**: Dictionary mapping node IDs to nodes

##### `get_branch_points() -> List[np.ndarray]`

Get all branch points.

- **Returns**: List of branch point positions

##### `get_endpoints() -> List[np.ndarray]`

Get all endpoints.

- **Returns**: List of endpoint positions

##### `calculate_total_length() -> float`

Calculate the total length of all segments.

- **Returns**: The total length

##### `to_dict() -> Dict[str, Any]`

Convert the model to a dictionary.

- **Returns**: Dictionary representation of the model

##### `save_to_json(filename: str) -> None`

Save the model to a JSON file.

- **Parameters**:
  - `filename`: The filename to save to

### VesselSegment

Class for representing a vessel segment.

```python
from src.utils.data_structures import VesselSegment
```

#### Methods

##### `__init__(segment_id: int, points: np.ndarray, start_node_id: str = None, end_node_id: str = None)`

Initialize a vessel segment.

- **Parameters**:
  - `segment_id`: The ID of the segment
  - `points`: The points defining the segment
  - `start_node_id`: The ID of the start node
  - `end_node_id`: The ID of the end node

##### `set_property(key: str, value: Any) -> None`

Set a property of the segment.

- **Parameters**:
  - `key`: The property key
  - `value`: The property value

##### `get_property(key: str, default: Any = None) -> Any`

Get a property of the segment.

- **Parameters**:
  - `key`: The property key
  - `default`: The default value to return if the property does not exist
- **Returns**: The property value

##### `calculate_length() -> float`

Calculate the length of the segment.

- **Returns**: The length of the segment

##### `get_point_at_distance(distance: float) -> np.ndarray`

Get a point at a specific distance along the segment.

- **Parameters**:
  - `distance`: The distance along the segment
- **Returns**: The point at the specified distance

##### `get_direction_at_distance(distance: float) -> np.ndarray`

Get the direction at a specific distance along the segment.

- **Parameters**:
  - `distance`: The distance along the segment
- **Returns**: The direction at the specified distance

##### `to_dict() -> Dict[str, Any]`

Convert the segment to a dictionary.

- **Returns**: Dictionary representation of the segment

### VesselNode

Class for representing a vessel node (branch point or endpoint).

```python
from src.utils.data_structures import VesselNode
```

#### Methods

##### `__init__(node_id: str, position: np.ndarray, node_type: str = "unknown")`

Initialize a vessel node.

- **Parameters**:
  - `node_id`: The ID of the node
  - `position`: The position of the node
  - `node_type`: The type of the node (e.g., "branch", "endpoint")

##### `set_property(key: str, value: Any) -> None`

Set a property of the node.

- **Parameters**:
  - `key`: The property key
  - `value`: The property value

##### `get_property(key: str, default: Any = None) -> Any`

Get a property of the node.

- **Parameters**:
  - `key`: The property key
  - `default`: The default value to return if the property does not exist
- **Returns**: The property value

##### `add_connected_segment(segment_id: int) -> None`

Add a connected segment to the node.

- **Parameters**:
  - `segment_id`: The ID of the connected segment

##### `get_connected_segments() -> List[int]`

Get the connected segments of the node.

- **Returns**: The IDs of the connected segments

##### `to_dict() -> Dict[str, Any]`

Convert the node to a dictionary.

- **Returns**: Dictionary representation of the node
