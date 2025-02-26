"""
Script to create sample STL files for testing the blood vessel analysis system.
"""

import os
import numpy as np
from stl import mesh

def create_tube_stl(filename, radius=1.0, length=10.0, num_segments=20):
    """
    Create a simple tube STL file.
    
    Args:
        filename: Path to save the STL file.
        radius: Radius of the tube.
        length: Length of the tube.
        num_segments: Number of segments around the circumference.
    """
    # Create vertices for the tube
    vertices = []
    
    # Create points around the circumference at both ends of the tube
    for z in [0, length]:
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            vertices.append([x, y, z])
    
    # Create faces for the tube
    faces = []
    
    # Create the side faces
    for i in range(num_segments):
        i1 = i
        i2 = (i + 1) % num_segments
        i3 = i + num_segments
        i4 = ((i + 1) % num_segments) + num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])
    
    # Create the end cap faces
    # Bottom cap
    for i in range(1, num_segments - 1):
        faces.append([0, i, i + 1])
    
    # Top cap
    for i in range(1, num_segments - 1):
        faces.append([num_segments, num_segments + i + 1, num_segments + i])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    tube_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    
    # Set the vertices of each face
    for i, face in enumerate(faces):
        for j in range(3):
            tube_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to an STL file
    tube_mesh.save(filename)
    print(f"Created tube STL file: {filename}")

def create_y_vessel_stl(filename, radius=1.0, length=10.0, branch_angle=30.0, num_segments=20):
    """
    Create a Y-shaped vessel STL file.
    
    Args:
        filename: Path to save the STL file.
        radius: Radius of the vessel.
        length: Length of the main vessel.
        branch_angle: Angle of the branches in degrees.
        num_segments: Number of segments around the circumference.
    """
    # Create vertices for the Y-shaped vessel
    vertices = []
    
    # Create points around the circumference at the start of the main vessel
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        vertices.append([x, y, z])
    
    # Create points around the circumference at the branch point
    branch_point_z = length / 2
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = branch_point_z
        vertices.append([x, y, z])
    
    # Create points around the circumference at the end of the left branch
    branch_length = length / 2
    branch_angle_rad = np.radians(branch_angle)
    left_branch_x = -branch_length * np.sin(branch_angle_rad)
    left_branch_z = branch_point_z + branch_length * np.cos(branch_angle_rad)
    
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = left_branch_x + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = left_branch_z
        vertices.append([x, y, z])
    
    # Create points around the circumference at the end of the right branch
    right_branch_x = branch_length * np.sin(branch_angle_rad)
    right_branch_z = branch_point_z + branch_length * np.cos(branch_angle_rad)
    
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = right_branch_x + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = right_branch_z
        vertices.append([x, y, z])
    
    # Create faces for the Y-shaped vessel
    faces = []
    
    # Create the side faces for the main vessel
    for i in range(num_segments):
        i1 = i
        i2 = (i + 1) % num_segments
        i3 = i + num_segments
        i4 = ((i + 1) % num_segments) + num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])
    
    # Create the side faces for the left branch
    for i in range(num_segments):
        i1 = i + num_segments
        i2 = ((i + 1) % num_segments) + num_segments
        i3 = i + 2 * num_segments
        i4 = ((i + 1) % num_segments) + 2 * num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])
    
    # Create the side faces for the right branch
    for i in range(num_segments):
        i1 = i + num_segments
        i2 = ((i + 1) % num_segments) + num_segments
        i3 = i + 3 * num_segments
        i4 = ((i + 1) % num_segments) + 3 * num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i3, i2])
        faces.append([i2, i3, i4])
    
    # Create the end cap faces
    # Bottom cap
    for i in range(1, num_segments - 1):
        faces.append([0, i, i + 1])
    
    # Left branch cap
    for i in range(1, num_segments - 1):
        faces.append([2 * num_segments, 2 * num_segments + i + 1, 2 * num_segments + i])
    
    # Right branch cap
    for i in range(1, num_segments - 1):
        faces.append([3 * num_segments, 3 * num_segments + i, 3 * num_segments + i + 1])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    y_vessel_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    
    # Set the vertices of each face
    for i, face in enumerate(faces):
        for j in range(3):
            y_vessel_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to an STL file
    y_vessel_mesh.save(filename)
    print(f"Created Y-shaped vessel STL file: {filename}")

def create_bifurcation_stl(filename, radius=1.0, length=10.0, branch_angle=30.0, num_segments=20):
    """
    Create a bifurcation vessel STL file.
    
    Args:
        filename: Path to save the STL file.
        radius: Radius of the vessel.
        length: Length of the main vessel.
        branch_angle: Angle of the branches in degrees.
        num_segments: Number of segments around the circumference.
    """
    # Create vertices for the bifurcation vessel
    vertices = []
    
    # Create points around the circumference at the start of the main vessel
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = 0
        vertices.append([x, y, z])
    
    # Create points around the circumference at the branch point
    branch_point_z = length / 2
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = branch_point_z
        vertices.append([x, y, z])
    
    # Create points around the circumference at the end of the left branch
    branch_length = length / 2
    branch_angle_rad = np.radians(branch_angle)
    left_branch_x = -branch_length * np.sin(branch_angle_rad)
    left_branch_z = branch_point_z + branch_length * np.cos(branch_angle_rad)
    
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = left_branch_x + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = left_branch_z
        vertices.append([x, y, z])
    
    # Create points around the circumference at the end of the right branch
    right_branch_x = branch_length * np.sin(branch_angle_rad)
    right_branch_z = branch_point_z + branch_length * np.cos(branch_angle_rad)
    
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = right_branch_x + radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = right_branch_z
        vertices.append([x, y, z])
    
    # Create points around the circumference at the end of the main branch
    main_branch_z = branch_point_z + branch_length
    
    for i in range(num_segments):
        angle = 2 * np.pi * i / num_segments
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = main_branch_z
        vertices.append([x, y, z])
    
    # Create faces for the bifurcation vessel
    faces = []
    
    # Create the side faces for the main vessel (before branch)
    for i in range(num_segments):
        i1 = i
        i2 = (i + 1) % num_segments
        i3 = i + num_segments
        i4 = ((i + 1) % num_segments) + num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])
    
    # Create the side faces for the left branch
    for i in range(num_segments):
        i1 = i + num_segments
        i2 = ((i + 1) % num_segments) + num_segments
        i3 = i + 2 * num_segments
        i4 = ((i + 1) % num_segments) + 2 * num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i2, i3])
        faces.append([i2, i4, i3])
    
    # Create the side faces for the right branch
    for i in range(num_segments):
        i1 = i + num_segments
        i2 = ((i + 1) % num_segments) + num_segments
        i3 = i + 3 * num_segments
        i4 = ((i + 1) % num_segments) + 3 * num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i3, i2])
        faces.append([i2, i3, i4])
    
    # Create the side faces for the main branch (after branch)
    for i in range(num_segments):
        i1 = i + num_segments
        i2 = ((i + 1) % num_segments) + num_segments
        i3 = i + 4 * num_segments
        i4 = ((i + 1) % num_segments) + 4 * num_segments
        
        # Add two triangles for each side face
        faces.append([i1, i3, i2])
        faces.append([i2, i3, i4])
    
    # Create the end cap faces
    # Bottom cap
    for i in range(1, num_segments - 1):
        faces.append([0, i, i + 1])
    
    # Left branch cap
    for i in range(1, num_segments - 1):
        faces.append([2 * num_segments, 2 * num_segments + i + 1, 2 * num_segments + i])
    
    # Right branch cap
    for i in range(1, num_segments - 1):
        faces.append([3 * num_segments, 3 * num_segments + i, 3 * num_segments + i + 1])
    
    # Main branch cap
    for i in range(1, num_segments - 1):
        faces.append([4 * num_segments, 4 * num_segments + i, 4 * num_segments + i + 1])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    bifurcation_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    
    # Set the vertices of each face
    for i, face in enumerate(faces):
        for j in range(3):
            bifurcation_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to an STL file
    bifurcation_mesh.save(filename)
    print(f"Created bifurcation vessel STL file: {filename}")

def create_spiral_stl(filename, radius=1.0, length=10.0, num_turns=2, num_segments=20, num_points=50):
    """
    Create a spiral vessel STL file.
    
    Args:
        filename: Path to save the STL file.
        radius: Radius of the vessel.
        length: Length of the spiral.
        num_turns: Number of turns in the spiral.
        num_segments: Number of segments around the circumference.
        num_points: Number of points along the spiral.
    """
    # Create the centerline of the spiral
    centerline = []
    for i in range(num_points):
        t = i / (num_points - 1)
        angle = 2 * np.pi * num_turns * t
        x = t * length * np.cos(angle)
        y = t * length * np.sin(angle)
        z = t * length
        centerline.append([x, y, z])
    
    # Create vertices for the spiral vessel
    vertices = []
    
    # Create points around the circumference at each point along the centerline
    for point in centerline:
        # Calculate the tangent vector at this point
        if point is centerline[0]:
            tangent = np.array(centerline[1]) - np.array(point)
        elif point is centerline[-1]:
            tangent = np.array(point) - np.array(centerline[-2])
        else:
            idx = centerline.index(point)
            tangent = np.array(centerline[idx + 1]) - np.array(centerline[idx - 1])
        
        tangent = tangent / np.linalg.norm(tangent)
        
        # Calculate a perpendicular vector
        if abs(tangent[0]) < abs(tangent[1]):
            perp = np.array([1, 0, 0])
        else:
            perp = np.array([0, 1, 0])
        
        # Calculate the normal and binormal vectors
        normal = np.cross(tangent, perp)
        normal = normal / np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        
        # Create points around the circumference
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            x = point[0] + radius * (normal[0] * np.cos(angle) + binormal[0] * np.sin(angle))
            y = point[1] + radius * (normal[1] * np.cos(angle) + binormal[1] * np.sin(angle))
            z = point[2] + radius * (normal[2] * np.cos(angle) + binormal[2] * np.sin(angle))
            vertices.append([x, y, z])
    
    # Create faces for the spiral vessel
    faces = []
    
    # Create the side faces
    for i in range(num_points - 1):
        for j in range(num_segments):
            i1 = i * num_segments + j
            i2 = i * num_segments + (j + 1) % num_segments
            i3 = (i + 1) * num_segments + j
            i4 = (i + 1) * num_segments + (j + 1) % num_segments
            
            # Add two triangles for each side face
            faces.append([i1, i2, i3])
            faces.append([i2, i4, i3])
    
    # Create the end cap faces
    # Bottom cap
    for i in range(1, num_segments - 1):
        faces.append([0, i, i + 1])
    
    # Top cap
    top_start = (num_points - 1) * num_segments
    for i in range(1, num_segments - 1):
        faces.append([top_start, top_start + i + 1, top_start + i])
    
    # Create the mesh
    vertices = np.array(vertices)
    faces = np.array(faces)
    
    # Create the mesh
    spiral_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    
    # Set the vertices of each face
    for i, face in enumerate(faces):
        for j in range(3):
            spiral_mesh.vectors[i][j] = vertices[face[j]]
    
    # Save the mesh to an STL file
    spiral_mesh.save(filename)
    print(f"Created spiral vessel STL file: {filename}")

def main():
    """Create sample STL files."""
    # Create the sample_stl directory if it doesn't exist
    os.makedirs("sample_stl", exist_ok=True)
    
    # Create a simple tube
    create_tube_stl("sample_stl/tube.stl", radius=1.0, length=10.0, num_segments=20)
    
    # Create a Y-shaped vessel
    create_y_vessel_stl("sample_stl/y_vessel.stl", radius=1.0, length=10.0, branch_angle=30.0, num_segments=20)
    
    # Create a bifurcation vessel
    create_bifurcation_stl("sample_stl/bifurcation.stl", radius=1.0, length=10.0, branch_angle=30.0, num_segments=20)
    
    # Create a spiral vessel
    create_spiral_stl("sample_stl/spiral.stl", radius=0.5, length=10.0, num_turns=2, num_segments=20, num_points=50)

if __name__ == "__main__":
    main()
