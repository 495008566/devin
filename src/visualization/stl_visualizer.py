"""
STL Visualization module.

This module provides functionality for visualizing STL blood vessel models.
"""

import numpy as np
import pyvista as pv
from typing import Dict, Any, List, Tuple, Optional, Union
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tempfile


class STLVisualizer:
    """Class for visualizing STL blood vessel models."""

    def __init__(self):
        """Initialize the STL visualizer."""
        self.mesh = None
        self.centerline = None
        self.branch_points = None
        self.endpoints = None
        self.segments = None
        self.cross_sections = None
        self.plotter = None

    def set_mesh(self, mesh):
        """
        Set the mesh to visualize.

        Args:
            mesh: The mesh to visualize.
        """
        if hasattr(mesh, 'vectors'):
            # Convert numpy-stl mesh to PyVista mesh
            vertices = mesh.vectors.reshape(-1, 3)
            faces = np.arange(len(vertices)).reshape(-1, 3)
            self.mesh = pv.PolyData(vertices, faces)
        else:
            # Assume it's already a PyVista mesh or can be converted
            self.mesh = pv.wrap(mesh)

    def set_centerline(self, centerline: np.ndarray):
        """
        Set the centerline to visualize.

        Args:
            centerline: The centerline points.
        """
        self.centerline = centerline

    def set_branch_points(self, branch_points: np.ndarray):
        """
        Set the branch points to visualize.

        Args:
            branch_points: The branch points.
        """
        self.branch_points = branch_points

    def set_endpoints(self, endpoints: np.ndarray):
        """
        Set the endpoints to visualize.

        Args:
            endpoints: The endpoints.
        """
        self.endpoints = endpoints

    def set_segments(self, segments: List[np.ndarray]):
        """
        Set the segments to visualize.

        Args:
            segments: The centerline segments.
        """
        self.segments = segments

    def set_cross_sections(self, cross_sections: List[Dict[str, Any]]):
        """
        Set the cross-sections to visualize.

        Args:
            cross_sections: The cross-sections.
        """
        self.cross_sections = cross_sections

    def create_plotter(self, off_screen: bool = False, window_size: Tuple[int, int] = (1024, 768)):
        """
        Create a PyVista plotter.

        Args:
            off_screen: Whether to render off-screen.
            window_size: The window size.

        Returns:
            pv.Plotter: The created plotter.
        """
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=window_size)
        return self.plotter

    def visualize_mesh(self, color: str = 'red', opacity: float = 0.5, show_edges: bool = False):
        """
        Visualize the mesh.

        Args:
            color: The color of the mesh.
            opacity: The opacity of the mesh.
            show_edges: Whether to show the mesh edges.
        """
        if self.mesh is None:
            print("Error: No mesh to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        self.plotter.add_mesh(self.mesh, color=color, opacity=opacity, show_edges=show_edges)

    def visualize_centerline(self, color: str = 'blue', line_width: float = 5.0):
        """
        Visualize the centerline.

        Args:
            color: The color of the centerline.
            line_width: The width of the centerline.
        """
        if self.centerline is None or len(self.centerline) < 2:
            print("Error: No centerline to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Create a PyVista line from the centerline points
        line = pv.Line(self.centerline[0], self.centerline[-1], resolution=len(self.centerline) - 1)
        line.points = self.centerline

        self.plotter.add_mesh(line, color=color, line_width=line_width)

    def visualize_segments(self, colors: Union[str, List[str]] = None, line_width: float = 3.0):
        """
        Visualize the centerline segments.

        Args:
            colors: The color(s) of the segments.
            line_width: The width of the segments.
        """
        if self.segments is None or not self.segments:
            print("Error: No segments to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Generate colors if not provided
        if colors is None:
            # Create a colormap
            cmap = plt.cm.get_cmap('tab10', len(self.segments))
            colors = [cmap(i)[:3] for i in range(len(self.segments))]
        elif isinstance(colors, str):
            colors = [colors] * len(self.segments)

        # Add each segment as a separate line
        for i, segment in enumerate(self.segments):
            if len(segment) < 2:
                continue

            # Create a PyVista line from the segment points
            line = pv.Line(segment[0], segment[-1], resolution=len(segment) - 1)
            line.points = segment

            self.plotter.add_mesh(line, color=colors[i % len(colors)], line_width=line_width)

    def visualize_branch_points(self, color: str = 'green', point_size: float = 15.0):
        """
        Visualize the branch points.

        Args:
            color: The color of the branch points.
            point_size: The size of the branch points.
        """
        if self.branch_points is None or len(self.branch_points) == 0:
            print("Error: No branch points to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Create a PyVista point cloud from the branch points
        points = pv.PolyData(self.branch_points)

        self.plotter.add_mesh(points, color=color, point_size=point_size, render_points_as_spheres=True)

    def visualize_endpoints(self, color: str = 'yellow', point_size: float = 15.0):
        """
        Visualize the endpoints.

        Args:
            color: The color of the endpoints.
            point_size: The size of the endpoints.
        """
        if self.endpoints is None or len(self.endpoints) == 0:
            print("Error: No endpoints to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Create a PyVista point cloud from the endpoints
        points = pv.PolyData(self.endpoints)

        self.plotter.add_mesh(points, color=color, point_size=point_size, render_points_as_spheres=True)

    def visualize_cross_sections(self, color: str = 'cyan', opacity: float = 0.7):
        """
        Visualize the cross-sections.

        Args:
            color: The color of the cross-sections.
            opacity: The opacity of the cross-sections.
        """
        if self.cross_sections is None or not self.cross_sections:
            print("Error: No cross-sections to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Add each cross-section as a separate mesh
        for cs in self.cross_sections:
            if "error" in cs or "vertices_3d" not in cs:
                continue

            vertices = cs["vertices_3d"]
            if len(vertices) < 3:
                continue

            # Create a PyVista polygon from the vertices
            polygon = pv.PolyData(vertices)
            
            # Create faces for the polygon
            n_points = len(vertices)
            faces = np.ones((1, n_points + 1), dtype=np.int64)
            faces[0, 0] = n_points
            faces[0, 1:] = np.arange(n_points)
            polygon.faces = faces.flatten()

            self.plotter.add_mesh(polygon, color=color, opacity=opacity)

    def visualize_all(self, mesh_color: str = 'red', mesh_opacity: float = 0.3,
                     centerline_color: str = 'blue', segment_colors: List[str] = None,
                     branch_color: str = 'green', endpoint_color: str = 'yellow',
                     cross_section_color: str = 'cyan', cross_section_opacity: float = 0.7):
        """
        Visualize all components.

        Args:
            mesh_color: The color of the mesh.
            mesh_opacity: The opacity of the mesh.
            centerline_color: The color of the centerline.
            segment_colors: The colors of the segments.
            branch_color: The color of the branch points.
            endpoint_color: The color of the endpoints.
            cross_section_color: The color of the cross-sections.
            cross_section_opacity: The opacity of the cross-sections.
        """
        if self.plotter is None:
            self.create_plotter()

        # Visualize the mesh
        if self.mesh is not None:
            self.visualize_mesh(color=mesh_color, opacity=mesh_opacity)

        # Visualize the segments
        if self.segments is not None and self.segments:
            self.visualize_segments(colors=segment_colors)

        # Visualize the branch points
        if self.branch_points is not None and len(self.branch_points) > 0:
            self.visualize_branch_points(color=branch_color)

        # Visualize the endpoints
        if self.endpoints is not None and len(self.endpoints) > 0:
            self.visualize_endpoints(color=endpoint_color)

        # Visualize the cross-sections
        if self.cross_sections is not None and self.cross_sections:
            self.visualize_cross_sections(color=cross_section_color, opacity=cross_section_opacity)

    def add_scalar_bar(self, title: str = ""):
        """
        Add a scalar bar to the visualization.

        Args:
            title: The title of the scalar bar.
        """
        if self.plotter is None:
            print("Error: No plotter created.")
            return

        self.plotter.add_scalar_bar(title=title)

    def add_axes(self):
        """Add axes to the visualization."""
        if self.plotter is None:
            print("Error: No plotter created.")
            return

        self.plotter.add_axes()

    def add_legend(self, labels: List[str], colors: List[str]):
        """
        Add a legend to the visualization.

        Args:
            labels: The labels for the legend.
            colors: The colors for the legend.
        """
        if self.plotter is None:
            print("Error: No plotter created.")
            return

        if len(labels) != len(colors):
            print("Error: Number of labels must match number of colors.")
            return

        self.plotter.add_legend(labels, colors)

    def show(self, interactive: bool = True, screenshot: str = None):
        """
        Show the visualization.

        Args:
            interactive: Whether to show the visualization interactively.
            screenshot: Path to save a screenshot, or None to not save.

        Returns:
            Optional[np.ndarray]: The screenshot image array if off_screen is True, None otherwise.
        """
        if self.plotter is None:
            print("Error: No plotter created.")
            return None

        if screenshot is not None:
            return self.plotter.screenshot(screenshot)
        else:
            return self.plotter.show(interactive=interactive)

    def close(self):
        """Close the visualization."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    def create_animation(self, filename: str, n_frames: int = 36, fps: int = 15,
                        orbit_axis: str = 'z', orbit_factor: float = 1.0):
        """
        Create an animation of the visualization.

        Args:
            filename: Path to save the animation.
            n_frames: Number of frames in the animation.
            fps: Frames per second.
            orbit_axis: Axis to orbit around ('x', 'y', or 'z').
            orbit_factor: Factor to control the orbit speed.

        Returns:
            str: Path to the saved animation.
        """
        if self.plotter is None:
            print("Error: No plotter created.")
            return None

        # Create a temporary directory to store the frames
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create the frames
            for i in range(n_frames):
                # Rotate the camera
                angle = i * 360.0 / n_frames * orbit_factor
                if orbit_axis.lower() == 'x':
                    self.plotter.camera.elevation = angle
                elif orbit_axis.lower() == 'y':
                    self.plotter.camera.roll = angle
                else:  # 'z'
                    self.plotter.camera.azimuth = angle

                # Save the frame
                frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                self.plotter.screenshot(frame_path)

            # Use PyVista's animation writer to create the animation
            try:
                import imageio
                frames = []
                for i in range(n_frames):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    frames.append(imageio.imread(frame_path))
                
                imageio.mimsave(filename, frames, fps=fps)
                return filename
            except ImportError:
                print("Error: imageio is required for creating animations.")
                return None

    def create_interactive_visualization(self, filename: str = None):
        """
        Create an interactive visualization.

        Args:
            filename: Path to save the visualization, or None to not save.

        Returns:
            Optional[str]: Path to the saved visualization, or None if not saved.
        """
        if self.plotter is None:
            print("Error: No plotter created.")
            return None

        if filename is None:
            # Just show the interactive visualization
            self.plotter.show(interactive=True)
            return None
        else:
            # Save the visualization as HTML
            try:
                self.plotter.export_html(filename)
                return filename
            except Exception as e:
                print(f"Error creating interactive visualization: {e}")
                return None

    def color_mesh_by_distance_from_centerline(self, cmap_name: str = 'jet'):
        """
        Color the mesh by distance from the centerline.

        Args:
            cmap_name: The name of the colormap to use.
        """
        if self.mesh is None:
            print("Error: No mesh to visualize.")
            return

        if self.centerline is None or len(self.centerline) < 2:
            print("Error: No centerline available.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Calculate the distance from each mesh point to the centerline
        distances = []
        for point in self.mesh.points:
            # Find the minimum distance to any centerline point
            min_dist = float('inf')
            for cl_point in self.centerline:
                dist = np.linalg.norm(point - cl_point)
                min_dist = min(min_dist, dist)
            distances.append(min_dist)

        # Add the distances as a scalar field to the mesh
        self.mesh['distance'] = distances

        # Visualize the mesh with the distance scalar field
        self.plotter.add_mesh(self.mesh, scalars='distance', cmap=cmap_name, show_scalar_bar=True)

    def color_mesh_by_curvature(self, curvature_type: str = 'mean', cmap_name: str = 'coolwarm'):
        """
        Color the mesh by curvature.

        Args:
            curvature_type: The type of curvature to use ('mean', 'gaussian', or 'principal').
            cmap_name: The name of the colormap to use.
        """
        if self.mesh is None:
            print("Error: No mesh to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Calculate the curvature
        if curvature_type.lower() == 'mean':
            self.mesh.compute_curvature('mean')
            scalars = 'Mean_Curvature'
        elif curvature_type.lower() == 'gaussian':
            self.mesh.compute_curvature('gaussian')
            scalars = 'Gaussian_Curvature'
        else:  # 'principal'
            self.mesh.compute_curvature('principal')
            scalars = 'Principal_Curvature'

        # Visualize the mesh with the curvature scalar field
        self.plotter.add_mesh(self.mesh, scalars=scalars, cmap=cmap_name, show_scalar_bar=True)

    def visualize_cross_section_properties(self, property_name: str = 'area', cmap_name: str = 'viridis'):
        """
        Visualize cross-section properties.

        Args:
            property_name: The name of the property to visualize.
            cmap_name: The name of the colormap to use.
        """
        if self.cross_sections is None or not self.cross_sections:
            print("Error: No cross-sections to visualize.")
            return

        if self.plotter is None:
            self.create_plotter()

        # Extract the property values
        values = []
        positions = []
        for cs in self.cross_sections:
            if "error" in cs or property_name not in cs:
                continue
            values.append(cs[property_name])
            positions.append(cs.get("position", [0, 0, 0]))

        if not values:
            print(f"Error: No '{property_name}' property found in cross-sections.")
            return

        # Create a colormap
        cmap = plt.cm.get_cmap(cmap_name)
        norm = plt.Normalize(min(values), max(values))
        colors = [cmap(norm(value)) for value in values]

        # Add each cross-section as a separate mesh with the property-based color
        for i, cs in enumerate(self.cross_sections):
            if "error" in cs or "vertices_3d" not in cs or i >= len(colors):
                continue

            vertices = cs["vertices_3d"]
            if len(vertices) < 3:
                continue

            # Create a PyVista polygon from the vertices
            polygon = pv.PolyData(vertices)
            
            # Create faces for the polygon
            n_points = len(vertices)
            faces = np.ones((1, n_points + 1), dtype=np.int64)
            faces[0, 0] = n_points
            faces[0, 1:] = np.arange(n_points)
            polygon.faces = faces.flatten()

            self.plotter.add_mesh(polygon, color=colors[i][:3])

        # Add a scalar bar
        self.plotter.add_scalar_bar(title=property_name)
