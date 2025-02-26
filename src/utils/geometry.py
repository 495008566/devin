"""
Geometry utility functions.

This module provides utility functions for geometric calculations.
"""

import numpy as np
from typing import List, Tuple, Optional


def calculate_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1: The first point.
        point2: The second point.

    Returns:
        float: The Euclidean distance between the points.
    """
    return np.linalg.norm(point2 - point1)


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in degrees.

    Args:
        v1: The first vector.
        v2: The second vector.

    Returns:
        float: The angle between the vectors in degrees.
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    dot_product = np.dot(v1_norm, v2_norm)
    # Clamp to [-1, 1] to avoid numerical issues
    dot_product = max(-1.0, min(1.0, dot_product))
    angle_rad = np.arccos(dot_product)
    angle_deg = angle_rad * 180 / np.pi
    return angle_deg


def calculate_plane_normal(points: np.ndarray) -> np.ndarray:
    """
    Calculate the normal vector of a plane defined by three or more points.

    Args:
        points: Array of points defining the plane.

    Returns:
        np.ndarray: The normal vector of the plane.
    """
    if len(points) < 3:
        raise ValueError("At least three points are required to define a plane.")
    
    # Use the first three points to define the plane
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    
    # Calculate the normal vector using the cross product
    normal = np.cross(v1, v2)
    
    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)
    
    return normal


def project_point_to_line(point: np.ndarray, line_point: np.ndarray, line_direction: np.ndarray) -> np.ndarray:
    """
    Project a point onto a line.

    Args:
        point: The point to project.
        line_point: A point on the line.
        line_direction: The direction vector of the line.

    Returns:
        np.ndarray: The projected point.
    """
    # Normalize the line direction
    line_direction = line_direction / np.linalg.norm(line_direction)
    
    # Calculate the vector from the line point to the point
    v = point - line_point
    
    # Calculate the projection of v onto the line direction
    proj = np.dot(v, line_direction)
    
    # Calculate the projected point
    projected_point = line_point + proj * line_direction
    
    return projected_point


def project_point_to_plane(point: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    """
    Project a point onto a plane.

    Args:
        point: The point to project.
        plane_point: A point on the plane.
        plane_normal: The normal vector of the plane.

    Returns:
        np.ndarray: The projected point.
    """
    # Normalize the plane normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Calculate the vector from the plane point to the point
    v = point - plane_point
    
    # Calculate the projection of v onto the plane normal
    proj = np.dot(v, plane_normal)
    
    # Calculate the projected point
    projected_point = point - proj * plane_normal
    
    return projected_point


def calculate_curvature(points: np.ndarray, window_size: int = 5) -> List[float]:
    """
    Calculate the curvature at each point of a curve.

    Args:
        points: The curve points.
        window_size: The window size for curvature calculation.

    Returns:
        List[float]: The curvature at each point.
    """
    if len(points) < 3:
        return [0.0] * len(points)
    
    curvatures = []
    n = len(points)
    
    for i in range(n):
        # Define the window around the current point
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(n - 1, i + half_window)
        
        if end_idx - start_idx < 2:
            # Not enough points for curvature calculation
            curvatures.append(0.0)
            continue
        
        # Extract the window points
        window_points = points[start_idx:end_idx+1]
        
        # Fit a circle to the window points
        try:
            # Center the points
            centroid = np.mean(window_points, axis=0)
            centered_points = window_points - centroid
            
            # Construct the design matrix
            A = np.column_stack([
                centered_points[:, 0],
                centered_points[:, 1],
                np.ones(len(centered_points))
            ])
            
            # Construct the target vector
            b = centered_points[:, 0]**2 + centered_points[:, 1]**2
            
            # Solve the linear system
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Calculate the center and radius of the fitted circle
            center = np.array([x[0]/2, x[1]/2]) + centroid
            radius = np.sqrt(x[2] + (x[0]/2)**2 + (x[1]/2)**2)
            
            # Calculate the curvature (1/radius)
            curvature = 1.0 / radius if radius > 0 else 0.0
            curvatures.append(curvature)
        
        except np.linalg.LinAlgError:
            # Singular matrix, can't fit a circle
            curvatures.append(0.0)
    
    return curvatures


def calculate_torsion(points: np.ndarray, window_size: int = 7) -> List[float]:
    """
    Calculate the torsion at each point of a 3D curve.

    Args:
        points: The curve points.
        window_size: The window size for torsion calculation.

    Returns:
        List[float]: The torsion at each point.
    """
    if len(points) < 4 or points.shape[1] < 3:
        return [0.0] * len(points)
    
    torsions = []
    n = len(points)
    
    for i in range(n):
        # Define the window around the current point
        half_window = window_size // 2
        start_idx = max(0, i - half_window)
        end_idx = min(n - 1, i + half_window)
        
        if end_idx - start_idx < 3:
            # Not enough points for torsion calculation
            torsions.append(0.0)
            continue
        
        # Extract the window points
        window_points = points[start_idx:end_idx+1]
        
        try:
            # Calculate the first, second, and third derivatives
            # using finite differences
            
            # First derivative
            d1 = np.zeros_like(window_points)
            d1[1:-1] = (window_points[2:] - window_points[:-2]) / 2
            d1[0] = window_points[1] - window_points[0]
            d1[-1] = window_points[-1] - window_points[-2]
            
            # Second derivative
            d2 = np.zeros_like(window_points)
            d2[1:-1] = (window_points[2:] - 2 * window_points[1:-1] + window_points[:-2])
            d2[0] = window_points[2] - 2 * window_points[1] + window_points[0]
            d2[-1] = window_points[-1] - 2 * window_points[-2] + window_points[-3]
            
            # Third derivative
            d3 = np.zeros_like(window_points)
            d3[2:-2] = (window_points[4:] - 3 * window_points[3:-1] + 3 * window_points[2:-2] - window_points[1:-3]) / 2
            d3[0] = d3[2]  # Approximate
            d3[1] = d3[2]  # Approximate
            d3[-2] = d3[-3]  # Approximate
            d3[-1] = d3[-3]  # Approximate
            
            # Calculate torsion at the center of the window
            center_idx = len(window_points) // 2
            r1 = d1[center_idx]
            r2 = d2[center_idx]
            r3 = d3[center_idx]
            
            # Calculate the cross product of r1 and r2
            r1_cross_r2 = np.cross(r1, r2)
            
            # Calculate the magnitude of r1_cross_r2
            r1_cross_r2_norm = np.linalg.norm(r1_cross_r2)
            
            # Calculate the dot product of r1_cross_r2 and r3
            dot_product = np.dot(r1_cross_r2, r3)
            
            # Calculate torsion
            if r1_cross_r2_norm > 1e-10:
                torsion = dot_product / (r1_cross_r2_norm ** 2)
            else:
                torsion = 0.0
            
            torsions.append(torsion)
        
        except Exception:
            # Error in calculation
            torsions.append(0.0)
    
    return torsions


def fit_cylinder(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit a cylinder to a set of points.

    Args:
        points: The points to fit a cylinder to.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: The center point, axis direction, and radius of the cylinder.
    """
    if len(points) < 5:
        raise ValueError("At least 5 points are required to fit a cylinder.")
    
    # Initial guess for the cylinder axis
    # Use PCA to find the principal direction
    centered_points = points - np.mean(points, axis=0)
    _, _, vh = np.linalg.svd(centered_points)
    axis_direction = vh[0]
    
    # Normalize the axis direction
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    
    # Project the points onto a plane perpendicular to the axis
    # Choose a point on the axis as the center point
    center_point = np.mean(points, axis=0)
    
    # Project the points onto the plane
    projected_points = np.zeros_like(points)
    for i, point in enumerate(points):
        projected_points[i] = project_point_to_plane(point, center_point, axis_direction)
    
    # Fit a circle to the projected points
    # Center the projected points
    centroid = np.mean(projected_points, axis=0)
    centered_projected_points = projected_points - centroid
    
    # Construct the design matrix
    A = np.column_stack([
        centered_projected_points[:, 0],
        centered_projected_points[:, 1],
        np.ones(len(centered_projected_points))
    ])
    
    # Construct the target vector
    b = centered_projected_points[:, 0]**2 + centered_projected_points[:, 1]**2
    
    # Solve the linear system
    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    # Calculate the center and radius of the fitted circle
    circle_center = np.array([x[0]/2, x[1]/2, 0]) + centroid
    radius = np.sqrt(x[2] + (x[0]/2)**2 + (x[1]/2)**2)
    
    # Project the circle center onto the axis
    axis_point = project_point_to_line(circle_center, center_point, axis_direction)
    
    return axis_point, axis_direction, radius


def calculate_hausdorff_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    Calculate the Hausdorff distance between two sets of points.

    Args:
        points1: The first set of points.
        points2: The second set of points.

    Returns:
        float: The Hausdorff distance.
    """
    # Calculate the distance from each point in points1 to the closest point in points2
    distances1 = []
    for p1 in points1:
        min_dist = float('inf')
        for p2 in points2:
            dist = np.linalg.norm(p1 - p2)
            min_dist = min(min_dist, dist)
        distances1.append(min_dist)
    
    # Calculate the distance from each point in points2 to the closest point in points1
    distances2 = []
    for p2 in points2:
        min_dist = float('inf')
        for p1 in points1:
            dist = np.linalg.norm(p2 - p1)
            min_dist = min(min_dist, dist)
        distances2.append(min_dist)
    
    # The Hausdorff distance is the maximum of the two directed Hausdorff distances
    hausdorff_distance = max(max(distances1), max(distances2))
    
    return hausdorff_distance
