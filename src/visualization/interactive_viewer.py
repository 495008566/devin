"""
Interactive viewer module.

This module provides an interactive viewer for blood vessel models.
"""

import numpy as np
import pyvista as pv
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import tempfile
import time

from src.visualization.stl_visualizer import STLVisualizer


class InteractiveViewer:
    """Class for interactive viewing of blood vessel models."""

    def __init__(self):
        """Initialize the interactive viewer."""
        self.visualizer = STLVisualizer()
        self.callbacks = {}
        self.ui_elements = {}
        self.current_view = "default"
        self.views = {}
        self.properties = {}

    def set_mesh(self, mesh):
        """
        Set the mesh to visualize.

        Args:
            mesh: The mesh to visualize.
        """
        self.visualizer.set_mesh(mesh)

    def set_centerline(self, centerline: np.ndarray):
        """
        Set the centerline to visualize.

        Args:
            centerline: The centerline points.
        """
        self.visualizer.set_centerline(centerline)

    def set_branch_points(self, branch_points: np.ndarray):
        """
        Set the branch points to visualize.

        Args:
            branch_points: The branch points.
        """
        self.visualizer.set_branch_points(branch_points)

    def set_endpoints(self, endpoints: np.ndarray):
        """
        Set the endpoints to visualize.

        Args:
            endpoints: The endpoints.
        """
        self.visualizer.set_endpoints(endpoints)

    def set_segments(self, segments: List[np.ndarray]):
        """
        Set the segments to visualize.

        Args:
            segments: The centerline segments.
        """
        self.visualizer.set_segments(segments)

    def set_cross_sections(self, cross_sections: List[Dict[str, Any]]):
        """
        Set the cross-sections to visualize.

        Args:
            cross_sections: The cross-sections.
        """
        self.visualizer.set_cross_sections(cross_sections)

    def set_property(self, key: str, value: Any):
        """
        Set a property of the viewer.

        Args:
            key: The property key.
            value: The property value.
        """
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """
        Get a property of the viewer.

        Args:
            key: The property key.
            default: The default value to return if the property does not exist.

        Returns:
            Any: The property value.
        """
        return self.properties.get(key, default)

    def create_plotter(self, off_screen: bool = False, window_size: Tuple[int, int] = (1024, 768),
                      title: str = "Blood Vessel Viewer"):
        """
        Create a PyVista plotter.

        Args:
            off_screen: Whether to render off-screen.
            window_size: The window size.
            title: The window title.

        Returns:
            pv.Plotter: The created plotter.
        """
        # Create a plotter with a menu bar
        plotter = self.visualizer.create_plotter(off_screen=off_screen, window_size=window_size)
        plotter.window_size = window_size
        plotter.title = title
        
        return plotter

    def add_view(self, name: str, camera_position: Tuple[float, float, float],
                focal_point: Tuple[float, float, float] = None,
                view_up: Tuple[float, float, float] = None):
        """
        Add a predefined view.

        Args:
            name: The name of the view.
            camera_position: The camera position.
            focal_point: The focal point.
            view_up: The view up vector.
        """
        self.views[name] = {
            "camera_position": camera_position,
            "focal_point": focal_point,
            "view_up": view_up
        }

    def set_view(self, name: str):
        """
        Set the current view.

        Args:
            name: The name of the view.
        """
        if name not in self.views:
            print(f"Error: View '{name}' not found.")
            return

        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        view = self.views[name]
        self.visualizer.plotter.camera_position = view["camera_position"]
        
        if view["focal_point"] is not None:
            self.visualizer.plotter.camera.focal_point = view["focal_point"]
        
        if view["view_up"] is not None:
            self.visualizer.plotter.camera.up = view["view_up"]
        
        self.current_view = name

    def add_callback(self, name: str, callback: Callable):
        """
        Add a callback function.

        Args:
            name: The name of the callback.
            callback: The callback function.
        """
        self.callbacks[name] = callback

    def add_slider(self, name: str, value_range: Tuple[float, float], initial_value: float,
                  title: str, callback: Callable, pointa: Tuple[float, float] = (0.1, 0.1),
                  pointb: Tuple[float, float] = (0.4, 0.1)):
        """
        Add a slider to the viewer.

        Args:
            name: The name of the slider.
            value_range: The range of values for the slider.
            initial_value: The initial value of the slider.
            title: The title of the slider.
            callback: The callback function to call when the slider value changes.
            pointa: The starting point of the slider.
            pointb: The ending point of the slider.
        """
        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        # Add the callback
        self.add_callback(name, callback)

        # Add the slider
        slider = self.visualizer.plotter.add_slider_widget(
            callback=lambda value: self.callbacks[name](value),
            rng=value_range,
            value=initial_value,
            title=title,
            pointa=pointa,
            pointb=pointb
        )

        # Store the slider
        self.ui_elements[name] = slider

    def add_checkbox(self, name: str, value: bool, title: str, callback: Callable,
                    position: Tuple[float, float] = (0.1, 0.1)):
        """
        Add a checkbox to the viewer.

        Args:
            name: The name of the checkbox.
            value: The initial value of the checkbox.
            title: The title of the checkbox.
            callback: The callback function to call when the checkbox value changes.
            position: The position of the checkbox.
        """
        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        # Add the callback
        self.add_callback(name, callback)

        # Add the checkbox
        checkbox = self.visualizer.plotter.add_checkbox_button_widget(
            callback=lambda value: self.callbacks[name](value),
            value=value,
            position=position
        )

        # Store the checkbox
        self.ui_elements[name] = checkbox

    def add_button(self, name: str, title: str, callback: Callable,
                  position: Tuple[float, float] = (0.1, 0.1)):
        """
        Add a button to the viewer.

        Args:
            name: The name of the button.
            title: The title of the button.
            callback: The callback function to call when the button is clicked.
            position: The position of the button.
        """
        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        # Add the callback
        self.add_callback(name, callback)

        # Add the button
        button = self.visualizer.plotter.add_text_button_widget(
            callback=lambda: self.callbacks[name](),
            text=title,
            position=position
        )

        # Store the button
        self.ui_elements[name] = button

    def add_dropdown(self, name: str, options: List[str], callback: Callable,
                    position: Tuple[float, float] = (0.1, 0.1)):
        """
        Add a dropdown menu to the viewer.

        Args:
            name: The name of the dropdown.
            options: The options for the dropdown.
            callback: The callback function to call when an option is selected.
            position: The position of the dropdown.
        """
        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        # Add the callback
        self.add_callback(name, callback)

        # Add the dropdown
        dropdown = self.visualizer.plotter.add_drop_down_button_widget(
            callback=lambda option: self.callbacks[name](option),
            options=options,
            position=position
        )

        # Store the dropdown
        self.ui_elements[name] = dropdown

    def setup_default_ui(self):
        """Set up the default UI elements."""
        if self.visualizer.plotter is None:
            print("Error: No plotter created.")
            return

        # Add a slider for mesh opacity
        self.add_slider(
            name="mesh_opacity",
            value_range=(0.0, 1.0),
            initial_value=0.5,
            title="Mesh Opacity",
            callback=self._on_mesh_opacity_changed,
            pointa=(0.1, 0.1),
            pointb=(0.4, 0.1)
        )

        # Add checkboxes for visibility
        self.add_checkbox(
            name="show_mesh",
            value=True,
            title="Show Mesh",
            callback=self._on_show_mesh_changed,
            position=(0.1, 0.2)
        )

        self.add_checkbox(
            name="show_centerline",
            value=True,
            title="Show Centerline",
            callback=self._on_show_centerline_changed,
            position=(0.1, 0.25)
        )

        self.add_checkbox(
            name="show_branch_points",
            value=True,
            title="Show Branch Points",
            callback=self._on_show_branch_points_changed,
            position=(0.1, 0.3)
        )

        self.add_checkbox(
            name="show_endpoints",
            value=True,
            title="Show Endpoints",
            callback=self._on_show_endpoints_changed,
            position=(0.1, 0.35)
        )

        self.add_checkbox(
            name="show_cross_sections",
            value=True,
            title="Show Cross Sections",
            callback=self._on_show_cross_sections_changed,
            position=(0.1, 0.4)
        )

        # Add a dropdown for coloring options
        self.add_dropdown(
            name="color_by",
            options=["Default", "Distance from Centerline", "Curvature", "Cross Section Area"],
            callback=self._on_color_by_changed,
            position=(0.5, 0.1)
        )

        # Add a dropdown for view options
        self.add_dropdown(
            name="view",
            options=["Default"] + list(self.views.keys()),
            callback=self._on_view_changed,
            position=(0.5, 0.2)
        )

        # Add buttons for actions
        self.add_button(
            name="reset_camera",
            title="Reset Camera",
            callback=self._on_reset_camera,
            position=(0.5, 0.3)
        )

        self.add_button(
            name="take_screenshot",
            title="Take Screenshot",
            callback=self._on_take_screenshot,
            position=(0.5, 0.35)
        )

        self.add_button(
            name="create_animation",
            title="Create Animation",
            callback=self._on_create_animation,
            position=(0.5, 0.4)
        )

    def _on_mesh_opacity_changed(self, opacity: float):
        """
        Callback for when the mesh opacity is changed.

        Args:
            opacity: The new opacity value.
        """
        if self.visualizer.plotter is None:
            return

        # Update the mesh opacity
        for actor in self.visualizer.plotter.renderer.actors.values():
            if hasattr(actor, 'GetProperty'):
                actor.GetProperty().SetOpacity(opacity)

        # Refresh the renderer
        self.visualizer.plotter.render()

    def _on_show_mesh_changed(self, show: bool):
        """
        Callback for when the show mesh checkbox is changed.

        Args:
            show: Whether to show the mesh.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated visibility
        self._redraw_with_visibility(show_mesh=show)

    def _on_show_centerline_changed(self, show: bool):
        """
        Callback for when the show centerline checkbox is changed.

        Args:
            show: Whether to show the centerline.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated visibility
        self._redraw_with_visibility(show_centerline=show)

    def _on_show_branch_points_changed(self, show: bool):
        """
        Callback for when the show branch points checkbox is changed.

        Args:
            show: Whether to show the branch points.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated visibility
        self._redraw_with_visibility(show_branch_points=show)

    def _on_show_endpoints_changed(self, show: bool):
        """
        Callback for when the show endpoints checkbox is changed.

        Args:
            show: Whether to show the endpoints.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated visibility
        self._redraw_with_visibility(show_endpoints=show)

    def _on_show_cross_sections_changed(self, show: bool):
        """
        Callback for when the show cross sections checkbox is changed.

        Args:
            show: Whether to show the cross sections.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated visibility
        self._redraw_with_visibility(show_cross_sections=show)

    def _on_color_by_changed(self, option: str):
        """
        Callback for when the color by dropdown is changed.

        Args:
            option: The selected option.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Redraw everything with the updated coloring
        self._redraw_with_coloring(option)

    def _on_view_changed(self, view_name: str):
        """
        Callback for when the view dropdown is changed.

        Args:
            view_name: The name of the selected view.
        """
        if view_name == "Default":
            # Reset the camera
            self.visualizer.plotter.reset_camera()
        else:
            # Set the view
            self.set_view(view_name)

    def _on_reset_camera(self):
        """Callback for when the reset camera button is clicked."""
        if self.visualizer.plotter is None:
            return

        # Reset the camera
        self.visualizer.plotter.reset_camera()

    def _on_take_screenshot(self):
        """Callback for when the take screenshot button is clicked."""
        if self.visualizer.plotter is None:
            return

        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"

        # Take a screenshot
        self.visualizer.plotter.screenshot(filename)
        print(f"Screenshot saved to: {filename}")

    def _on_create_animation(self):
        """Callback for when the create animation button is clicked."""
        if self.visualizer.plotter is None:
            return

        # Create a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"animation_{timestamp}.gif"

        # Create an animation
        self.visualizer.create_animation(filename)
        print(f"Animation saved to: {filename}")

    def _redraw_with_visibility(self, show_mesh: bool = None, show_centerline: bool = None,
                              show_branch_points: bool = None, show_endpoints: bool = None,
                              show_cross_sections: bool = None):
        """
        Redraw the visualization with updated visibility settings.

        Args:
            show_mesh: Whether to show the mesh.
            show_centerline: Whether to show the centerline.
            show_branch_points: Whether to show the branch points.
            show_endpoints: Whether to show the endpoints.
            show_cross_sections: Whether to show the cross sections.
        """
        if self.visualizer.plotter is None:
            return

        # Get the current visibility settings
        current_show_mesh = self.get_property("show_mesh", True)
        current_show_centerline = self.get_property("show_centerline", True)
        current_show_branch_points = self.get_property("show_branch_points", True)
        current_show_endpoints = self.get_property("show_endpoints", True)
        current_show_cross_sections = self.get_property("show_cross_sections", True)

        # Update the visibility settings
        if show_mesh is not None:
            self.set_property("show_mesh", show_mesh)
        if show_centerline is not None:
            self.set_property("show_centerline", show_centerline)
        if show_branch_points is not None:
            self.set_property("show_branch_points", show_branch_points)
        if show_endpoints is not None:
            self.set_property("show_endpoints", show_endpoints)
        if show_cross_sections is not None:
            self.set_property("show_cross_sections", show_cross_sections)

        # Get the updated visibility settings
        show_mesh = self.get_property("show_mesh", True)
        show_centerline = self.get_property("show_centerline", True)
        show_branch_points = self.get_property("show_branch_points", True)
        show_endpoints = self.get_property("show_endpoints", True)
        show_cross_sections = self.get_property("show_cross_sections", True)

        # Redraw everything with the updated visibility
        if show_mesh and self.visualizer.mesh is not None:
            self.visualizer.visualize_mesh()

        if show_centerline and self.visualizer.centerline is not None:
            self.visualizer.visualize_centerline()

        if show_branch_points and self.visualizer.branch_points is not None:
            self.visualizer.visualize_branch_points()

        if show_endpoints and self.visualizer.endpoints is not None:
            self.visualizer.visualize_endpoints()

        if show_cross_sections and self.visualizer.cross_sections is not None:
            self.visualizer.visualize_cross_sections()

        # Add axes
        self.visualizer.add_axes()

    def _redraw_with_coloring(self, color_by: str):
        """
        Redraw the visualization with updated coloring settings.

        Args:
            color_by: The coloring option.
        """
        if self.visualizer.plotter is None:
            return

        # Clear the plotter
        self.visualizer.plotter.clear()

        # Get the visibility settings
        show_mesh = self.get_property("show_mesh", True)
        show_centerline = self.get_property("show_centerline", True)
        show_branch_points = self.get_property("show_branch_points", True)
        show_endpoints = self.get_property("show_endpoints", True)
        show_cross_sections = self.get_property("show_cross_sections", True)

        # Apply the coloring
        if color_by == "Default":
            # Use default coloring
            if show_mesh and self.visualizer.mesh is not None:
                self.visualizer.visualize_mesh()
        elif color_by == "Distance from Centerline":
            # Color by distance from centerline
            if show_mesh and self.visualizer.mesh is not None:
                self.visualizer.color_mesh_by_distance_from_centerline()
        elif color_by == "Curvature":
            # Color by curvature
            if show_mesh and self.visualizer.mesh is not None:
                self.visualizer.color_mesh_by_curvature()
        elif color_by == "Cross Section Area":
            # Color by cross section area
            if show_cross_sections and self.visualizer.cross_sections is not None:
                self.visualizer.visualize_cross_section_properties(property_name="area")

        # Add the other components
        if show_centerline and self.visualizer.centerline is not None:
            self.visualizer.visualize_centerline()

        if show_branch_points and self.visualizer.branch_points is not None:
            self.visualizer.visualize_branch_points()

        if show_endpoints and self.visualizer.endpoints is not None:
            self.visualizer.visualize_endpoints()

        if show_cross_sections and self.visualizer.cross_sections is not None and color_by != "Cross Section Area":
            self.visualizer.visualize_cross_sections()

        # Add axes
        self.visualizer.add_axes()

    def run(self, interactive: bool = True):
        """
        Run the interactive viewer.

        Args:
            interactive: Whether to run in interactive mode.
        """
        if self.visualizer.plotter is None:
            self.create_plotter()

        # Set up the default UI
        self.setup_default_ui()

        # Visualize everything
        self._redraw_with_visibility()

        # Show the plotter
        self.visualizer.show(interactive=interactive)

    def close(self):
        """Close the viewer."""
        if self.visualizer is not None:
            self.visualizer.close()
