import sys
import re
import json
import logging
import numpy as np
import math
import trimesh
from shapely.geometry import LineString
import pyproj
import shapely
import pyvista as pv
import click

from utilities import get_min_height_swissalti_service, define_bridge_parameters

# Setup logger
logger = logging.getLogger(__name__)

# Add a formatter that includes timestamp, level, and module
formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# Add INFO level logging to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.setLevel(logging.DEBUG)


def repair_and_boolean_operation(mesh1, mesh2, bool_operation="intersection"):
    try:
        mesh1.fix_normals()
        mesh1.fill_holes()
        trimesh.repair.fix_winding(mesh1)
        mesh1.fill_holes()
        assert mesh1.volume

        mesh2.fix_normals()
        mesh2.fill_holes()
        trimesh.repair.fix_winding(mesh2)
        mesh2.fill_holes()
        assert mesh2.volume

        bool_result = trimesh.boolean.boolean_manifold(
            [mesh1, mesh2], operation=bool_operation, check_volume=False
        )

        bool_result.fix_normals()
        bool_result.fill_holes()
        trimesh.repair.fix_winding(bool_result)
        trimesh.repair.fix_inversion(bool_result)
        bool_result.fill_holes()

        if not bool_result.volume:
            mesh1.export("error_mesh1.stl")
            mesh2.export("error_mesh2.stl")
            assert bool_result.volume, f"While doing a {bool_operation} with two meshes, I couldn't get a proper volume. check ./error_mesh?.stl" 

        return bool_result
    except Exception as e:
        logger.error(f"Failed to process {mesh1} and {mesh2}")
        logger.error(f"Exception is '{e}'")
        raise


def create_bridge_deck(
    line_swiss, deck_width_pair, min_elevation, bottom_shift_percentage=0
):
    """
    Create a bridge deck mesh that follows a path.

    Args:
        line_swiss: Array of 3D coordinates in Swiss coordinates defining the path
        deck_width_pair: pair of left and right width of the bridge deck at the top in meters
        min_elevation: Minimum elevation for the bottom of the deck
        bottom_shift_percentage: Percentage to increase/decrease the bottom width relative to the top width.
                                 Positive values make the bottom wider, negative values make it narrower.

    Returns:
        trimesh.Trimesh: A mesh representing the bridge deck
    """
    # Calculate bottom deck width based on the shift percentage    
    bottom_deck_width_pair = ( deck_width_pair[0] + (bottom_shift_percentage * deck_width_pair[0]),
                               deck_width_pair[1] + (bottom_shift_percentage * deck_width_pair[1]) )
    deck_width = deck_width_pair[0] + deck_width_pair[1]
    bottom_deck_width = bottom_deck_width_pair[0] + bottom_deck_width_pair[1]
    
    # Convert input line to numpy array if it's not already
    coords = np.array(line_swiss)
    
    # Compute tangent vectors
    tangents = np.diff(coords, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])  # Repeat last to match shape
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]  # Normalize

    # Compute perpendicular vectors in XY plane
    perp_vectors = np.column_stack(
        [-tangents[:, 1], tangents[:, 0], np.zeros_like(tangents[:, 0])]
    )

    # Number of cross-sections
    num_sections = len(coords)

    # Create vertices for all cross-sections
    all_vertices = []
    section_sizes = []

    for i in range(num_sections):
        # Create cross-section at this point
        center = coords[i]
        perp = perp_vectors[i]

        # Create points for this cross-section (left edge, center, right edge)
        left_edge = center - perp * deck_width_pair[1]
        right_edge = center + perp * deck_width_pair[0]

        # Add vertices for top of deck
        all_vertices.append(left_edge)
        all_vertices.append(center)
        all_vertices.append(right_edge)

        # Create points for bottom of deck (at min_elevation)
        bottom_left = center - perp * bottom_deck_width_pair[1]
        bottom_left[2] = min_elevation
        bottom_center = center.copy()
        bottom_center[2] = min_elevation
        bottom_right = center + perp * bottom_deck_width_pair[0]
        bottom_right[2] = min_elevation

        all_vertices.append(bottom_left)
        all_vertices.append(bottom_center)
        all_vertices.append(bottom_right)

        # Each cross-section has 6 vertices
        section_sizes.append(6)

    # Convert to numpy array
    all_vertices = np.array(all_vertices)

    # Create faces to connect the cross-sections
    all_faces = []

    for i in range(num_sections - 1):
        # Indices for current and next cross-section
        curr_start = i * 6
        next_start = (i + 1) * 6

        # Connect top faces (deck surface)
        # Left triangle
        all_faces.append([curr_start, curr_start + 1, next_start + 1])
        all_faces.append([curr_start, next_start + 1, next_start])

        # Right triangle
        all_faces.append([curr_start + 1, curr_start + 2, next_start + 2])
        all_faces.append([curr_start + 1, next_start + 2, next_start + 1])

        # Connect bottom faces
        # Left triangle
        all_faces.append([curr_start + 3, next_start + 4, curr_start + 4])
        all_faces.append([curr_start + 3, next_start + 3, next_start + 4])

        # Right triangle
        all_faces.append([curr_start + 4, next_start + 4, next_start + 5])
        all_faces.append([curr_start + 4, next_start + 5, curr_start + 5])

        # Connect side faces (left side)
        all_faces.append([curr_start, next_start, next_start + 3])
        all_faces.append([curr_start, next_start + 3, curr_start + 3])

        # Connect side faces (right side)
        all_faces.append([curr_start + 2, curr_start + 5, next_start + 5])
        all_faces.append([curr_start + 2, next_start + 5, next_start + 2])

        # Connect front and back faces if needed (for first and last section)
        if i == 0:
            # Front face
            all_faces.append([curr_start, curr_start + 3, curr_start + 4])
            all_faces.append([curr_start, curr_start + 4, curr_start + 1])
            all_faces.append([curr_start + 1, curr_start + 4, curr_start + 5])
            all_faces.append([curr_start + 1, curr_start + 5, curr_start + 2])

        if i == num_sections - 2:
            # Back face
            all_faces.append([next_start, next_start + 1, next_start + 4])
            all_faces.append([next_start, next_start + 4, next_start + 3])
            all_faces.append([next_start + 1, next_start + 2, next_start + 5])
            all_faces.append([next_start + 1, next_start + 5, next_start + 4])

    # Create the deck mesh
    return trimesh.Trimesh(vertices=all_vertices, faces=all_faces), coords, perp_vectors


def create_tapered_box(w, width, height):
    """
    Create a box where the bottom face is shorter than the top face in the x-axis.
    The box is centered at the origin like trimesh.creation.box.

    Parameters:
    -----------
    w : float
        Width of the top face along x-axis
    width : float
        Width along y-axis
    height : float
        Height along z-axis
    """

    # Calculate half-dimensions for centering
    w_half = w / 2
    width_half = width / 2
    height_half = height / 2

    # Calculate the bottom face width
    bottom_width = w
    bottom_half = bottom_width / 2

    # Define vertices - centered around origin
    vertices = np.array(
        [
            # Top face (z=+height_half)
            [-w_half, -width_half, height_half],  # 0
            [w_half, -width_half, height_half],  # 1
            [w_half, width_half, height_half],  # 2
            [-w_half, width_half, height_half],  # 3
            # Bottom face (z=-height_half)
            [-bottom_half, -width_half, -height_half],  # 4
            [bottom_half, -width_half, -height_half],  # 5
            [bottom_half, width_half, -height_half],  # 6
            [-bottom_half, width_half, -height_half],  # 7
        ]
    )

    # Define faces (using triangles)
    faces = np.array(
        [
            # Top face
            [0, 1, 2],
            [0, 2, 3],
            # Bottom face
            [4, 5, 6],
            [4, 6, 7],
            # Side faces
            [0, 4, 5],
            [0, 5, 1],  # Front face
            [1, 5, 6],
            [1, 6, 2],  # Right face
            [2, 6, 7],
            [2, 7, 3],  # Back face
            [3, 7, 4],
            [3, 4, 0],  # Left face
        ]
    )

    # Create the mesh
    tapered_box = trimesh.Trimesh(vertices=vertices, faces=faces)

    return tapered_box


def create_box_arch(
    start_point,
    end_point,
    deck_width,
    box_heights,
    min_elevation,
    start_direction=None,
    end_direction=None,
    bottom_shift_meter=0,
    circular_arch=True,
    arch_height=5,
):
    """
    Create a solid box arch between two 3D points with half-cylinders
    on the top face parallel to the path. The box will have front and rear faces
    perpendicular to the bridge direction at those points.

    Args:
        start_point: 3D coordinates of the starting point
        end_point: 3D coordinates of the ending point
        deck_width: Width of the deck in meters
        box_height: tuple of heights of the box from min_elevation (start and end point)
        min_elevation: Minimum elevation for the bottom of the box
        start_direction: Direction vector at the start point (if None, calculated from start to end)
        end_direction: Direction vector at the end point (if None, calculated from start to end)
        bottom_shift_meter: How much to shift the bottom of the arch inward
        circular_arch: Whether to use circular arches
        arch_height: What is the height from the deck to the highst part of an arch

    Returns:
        trimesh.Trimesh: A mesh representing the box arch, possibly with a circular arch on top
    """
    deck_width_max = deck_width * 5 # the arch we are building is a sbutraction
                                    # solid, this makes sure thatthe
                                    # subtraction solid is large enough to cut
                                    # through the bridge deck
    
    logger.debug(f"width={deck_width}")
    # Convert input points to numpy arrays if they're not already
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    distance_start_end = np.linalg.norm(
        end_point - start_point
    )  # Distance between start and end points

    # Calculate the main direction vector of the arch (from start to end)
    main_direction = end_point[:2] - start_point[:2]  # Only consider X and Y
    main_direction = main_direction / np.linalg.norm(main_direction)

    # Use provided directions or default to main direction
    if start_direction is None:
        start_direction = main_direction
    if end_direction is None:
        end_direction = main_direction

    # Ensure directions are 2D and normalized
    start_direction = np.array(start_direction[:2])
    start_direction = start_direction / np.linalg.norm(start_direction)
    end_direction = np.array(end_direction[:2])
    end_direction = end_direction / np.linalg.norm(end_direction)

    # Calculate perpendicular vectors to the directions (in XY plane)
    start_perp = np.array([-start_direction[1], start_direction[0], 0])
    end_perp = np.array([-end_direction[1], end_direction[0], 0])
    avg_perp = (start_perp + end_perp) / np.linalg.norm(start_perp + end_perp)

    # Calculate the corners of the box using the perpendicular vectors
    # This ensures front and rear faces perpendicular to the bridge direction

    # Top corners at start point (perpendicular to start direction)
    top_start_left = np.array(
        [
            start_point[0] - avg_perp[0] * (deck_width_max),
            start_point[1] - avg_perp[1] * (deck_width_max),
            min_elevation + box_heights[0],
        ]
    )
    top_start_right = np.array(
        [
            start_point[0] + avg_perp[0] * (deck_width_max),
            start_point[1] + avg_perp[1] * (deck_width_max),
            min_elevation + box_heights[0],
        ]
    )

    # Top corners at end point (perpendicular to end direction)
    top_end_left = np.array(
        [
            end_point[0] - avg_perp[0] * (deck_width_max),
            end_point[1] - avg_perp[1] * (deck_width_max),
            min_elevation + box_heights[1],
        ]
    )
    top_end_right = np.array(
        [
            end_point[0] + avg_perp[0] * (deck_width_max),
            end_point[1] + avg_perp[1] * (deck_width_max),
            min_elevation + box_heights[1],
        ]
    )

    # Bottom corners derived from top corners
    bottom_start_left = top_start_left.copy()
    bottom_start_left[2] = min_elevation

    bottom_start_right = top_start_right.copy()
    bottom_start_right[2] = min_elevation

    bottom_end_left = top_end_left.copy()
    bottom_end_left[2] = min_elevation

    bottom_end_right = top_end_right.copy()
    bottom_end_right[2] = min_elevation

    # Apply bottom shift if needed
    if bottom_shift_meter != 0:
        # Shift bottom corners inward at both ends
        bottom_start_left[0] += start_direction[0] * bottom_shift_meter
        bottom_start_left[1] += start_direction[1] * bottom_shift_meter
        bottom_start_right[0] += start_direction[0] * bottom_shift_meter
        bottom_start_right[1] += start_direction[1] * bottom_shift_meter

        bottom_end_left[0] -= end_direction[0] * bottom_shift_meter
        bottom_end_left[1] -= end_direction[1] * bottom_shift_meter
        bottom_end_right[0] -= end_direction[0] * bottom_shift_meter
        bottom_end_right[1] -= end_direction[1] * bottom_shift_meter

    # Add all vertices
    box_vertices = [
        bottom_start_left,
        bottom_start_right,
        bottom_end_right,
        bottom_end_left,  # Bottom face
        top_start_left,
        top_start_right,
        top_end_right,
        top_end_left,  # Top face
    ]

    # Create faces for the box (6 faces, 2 triangles each)
    box_faces = [
        # Bottom face
        [0, 1, 2],
        [0, 2, 3],
        # Top face
        [4, 7, 6],
        [4, 6, 5],
        # Side faces
        [0, 3, 7],
        [0, 7, 4],  # Left side
        [1, 5, 6],
        [1, 6, 2],  # Right side
        [0, 4, 5],
        [0, 5, 1],  # Start face
        [3, 2, 6],
        [3, 6, 7],  # End face
    ]

    # Create the box mesh
    box_mesh = trimesh.Trimesh(vertices=box_vertices, faces=box_faces)

    def calc_distance_center_top_face(w, h0):
        """returns distance in meters from center to top face. If it's zero, we will get
        a half-cylinder. If it's above zero, the radius of the cyclinder
         becomes bigger, but the actual arch becomes shorter"""
        d = ((w / 2) ** 2 - h0**2) / (2 * h0)
        if d < 0:
            return 0, w/2 # always ensure the arch is not larger than a cylinder
        else:
            return d, d + h0

    if circular_arch:
        # calculate the cone for the arch
        cone = None
        cone_segments = 100  # Number of segments for the cylinder

        distance_center_top_face, cylinder_radius = calc_distance_center_top_face(
            distance_start_end, arch_height
        )
        

        # Create a cylinder along the top face
        cone = trimesh.creation.cylinder(
            radius=cylinder_radius, height=deck_width_max, sections=cone_segments
        )

        # create the extrusion box
        extrusion_box = create_tapered_box(distance_start_end, deck_width_max, max(box_heights)*10)
        # translate along y-axis to cut-out only the cylinder part that is needed
        translation = np.eye(4)
        translation[1, 3] = deck_width_max*0.4 + distance_center_top_face
        extrusion_box.apply_transform(translation)

        try: 
            cone = repair_and_boolean_operation(
                cone, extrusion_box, bool_operation="intersection"
            )
        except AssertionError as e:
            logger.warning(f"it seems that the cone does not intersect with the box : {e}")
            

        # First rotate the cylinder to align with the Y axis
        # This puts the cylinder's axis along the Y axis
        rotation = np.eye(4)
        rotation[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        cone.apply_transform(rotation)

        # Now we need to rotate the cylinder to align with the main direction
        # We need to rotate from the Y axis to the main direction vector
        y_axis = np.array([0, 1, 0])
        rotation_axis = np.cross(
            y_axis, np.append(main_direction, 0)
        )  # Add Z=0 to main_direction

        # Only rotate if the rotation axis is not zero
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            rotation_angle = np.arccos(np.dot(y_axis, np.append(main_direction, 0)))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                rotation_angle, rotation_axis
            )
            cone.apply_transform(rotation_matrix)

        # Calculate the midpoint of the top face
        top_center_start = (top_start_left + top_start_right) / 2
        top_center_end = (top_end_left + top_end_right) / 2

        # Position the cylinder at the center of the top face
        midpoint = (top_center_start + top_center_end) / 2

        # The cylinder's center should be at the midpoint of the top face
        # But we need to adjust it to align the flat part with the top face
        midpoint_adjusted = midpoint.copy()
        midpoint_adjusted[2] -= distance_center_top_face

        translation = np.eye(4)
        translation[:3, 3] = midpoint_adjusted
        cone.apply_transform(translation)

        # Combine the box and half-cylinder
        # final_mesh = trimesh.util.concatenate([box_mesh, cylinder])
        final_mesh = repair_and_boolean_operation(
            cone, box_mesh, bool_operation="union"
        )
    else:
        final_mesh = box_mesh

    return final_mesh


def interpolate_point_at_distance(coords, target_distance):
    """
    Interpolate a point along a 3D linestring at a specific distance from the start.

    Args:
        coords: Array of 3D coordinates defining the path
        target_distance: Distance from the start of the path in meters

    Returns:
        numpy.ndarray: Interpolated 3D point
    """
    # Initialize distance counter
    current_distance = 0.0

    # Handle edge cases
    if target_distance <= 0:
        return coords[0]

    # Iterate through line segments
    for i in range(len(coords) - 1):
        start_point = coords[i]
        end_point = coords[i + 1]

        # Calculate segment length (XY plane only)
        segment_length = np.linalg.norm(end_point[:2] - start_point[:2])

        # Check if target distance is within this segment
        if current_distance + segment_length >= target_distance:
            # Calculate how far along this segment the target is
            segment_fraction = (target_distance - current_distance) / segment_length

            # Interpolate the point
            interpolated_point = start_point + segment_fraction * (
                end_point - start_point
            )
            return interpolated_point

        # Update distance counter
        current_distance += segment_length

    # If we've gone through all segments, return the last point
    return coords[-1]


def interpolate_direction_at_distance(coords, target_distance):
    """
    Calculate the direction vector along a 3D linestring at a specific distance from the start.

    Args:
        coords: Array of 3D coordinates defining the path
        target_distance: Distance from the start of the path in meters

    Returns:
        numpy.ndarray: Direction vector at the specified distance
    """
    # Initialize distance counter
    current_distance = 0.0

    # Handle edge cases
    if target_distance <= 0:
        # Return direction of first segment
        if len(coords) > 1:
            direction = coords[1][:2] - coords[0][:2]
            return direction / np.linalg.norm(direction)
        return np.array([1.0, 0.0])  # Default direction if only one point

    # Iterate through line segments
    for i in range(len(coords) - 1):
        start_point = coords[i]
        end_point = coords[i + 1]

        # Calculate segment length (XY plane only)
        segment_length = np.linalg.norm(end_point[:2] - start_point[:2])

        # Check if target distance is within this segment
        if current_distance + segment_length >= target_distance:
            # Use the direction of this segment
            direction = end_point[:2] - start_point[:2]
            return direction / np.linalg.norm(direction)

        # Update distance counter
        current_distance += segment_length

    # If we've gone through all segments, return the direction of the last segment
    if len(coords) > 1:
        direction = coords[-1][:2] - coords[-2][:2]
        return direction / np.linalg.norm(direction)

    return np.array([1.0, 0.0])  # Default direction if only one point


def create_arches(
    line_swiss,
    deck_width,
    min_elevation,
    arch_fractions,
    arch_height_fraction,
    pier_size_meters,
    bottom_shift_meter=0,
    circular_arch=True,
):
    """
    Create multiple box arches along a bridge path with gaps between them.
    The number and length of arches is determined by the provided fractions.
    All measurements are in meters.

    Args:
        line_swiss: Array of 3D coordinates in Swiss coordinates defining the path
        deck_width_: width of the deck in meters
        min_elevation: Minimum elevation for the bottom of the arches
        arch_fractions: List of fractions (summing to 1) that determine the length of each arch
        arch_height_fraction: fraction of the height for an arch (e.g. 0.8)
        pier_size_meters: Size of gaps between arches in meters
        bottom_shift_meter: How much to shift the bottom of the arch
        circular_arch: Whether to use circular arches

    Returns:
        list: List of trimesh.Trimesh objects representing the arches
    """
    # Convert input line to numpy array if it's not already    
    coords = np.array(line_swiss)

    # Calculate the total bridge length in meters
    bridge_length_meters = 0
    for i in range(len(coords) - 1):
        segment_length = np.linalg.norm(
            coords[i + 1][:2] - coords[i][:2]
        )  # XY distance only
        bridge_length_meters += segment_length

    logger.debug(f"Total bridge length: {bridge_length_meters:.2f} meters")

    # Validate the fractions
    if abs(sum(arch_fractions) - 1.0) > 0.001:
        logger.warning(
            f"Arch fractions sum to {sum(arch_fractions)}, not 1.0. Normalizing."
        )
        total = sum(arch_fractions)
        arch_fractions = [f / total for f in arch_fractions]

    num_arches = len(arch_fractions)
    logger.debug(f"Creating {num_arches} arches based on provided fractions")

    # Calculate the effective length available for arches (accounting for piers)
    # Each arch needs half a pier on each side, including the first and last arches
    total_pier_space = (
        pier_size_meters * (num_arches - 1) + pier_size_meters
    )  # Add one more pier for start and end
    effective_length = bridge_length_meters - total_pier_space

    arches = []
    current_distance = 0

    for i, fraction in enumerate(arch_fractions):
        # Calculate the length of this arch
        arch_length = fraction * effective_length

        # Calculate start and end distances for this arch
        if i == 0:
            # First arch starts after half a pier
            start_distance = pier_size_meters / 2
        else:
            # Other arches start after the previous arch plus a pier
            start_distance = current_distance + pier_size_meters

        end_distance = start_distance + arch_length
        current_distance = end_distance

        # Find the points at these distances
        start_point = interpolate_point_at_distance(coords, start_distance)
        end_point = interpolate_point_at_distance(coords, end_distance)

        # Calculate the height for the arch based on the average z-coordinate
        avg_z = (start_point[2] + end_point[2]) / 2
        average_bridge_height = avg_z - min_elevation
        box_height_start, box_height_end = (start_point[2] - min_elevation) * arch_height_fraction, \
            (end_point[2] - min_elevation) * arch_height_fraction
        delta_deck_arch = 4 # min meter between deck and the arch, should be
                            # based on the scale, let's assume 1:1000...
        arch_height = max(2, (average_bridge_height * (1- arch_height_fraction)) - delta_deck_arch)
        logger.debug(f"Arch height for arch {i}: {arch_height}")

        # Calculate the direction vectors at start and end points
        start_direction = interpolate_direction_at_distance(coords, start_distance)
        end_direction = interpolate_direction_at_distance(coords, end_distance)

        # Create the arch with perpendicular front and rear faces
        arch = create_box_arch(
            start_point=start_point,
            end_point=end_point,
            deck_width=deck_width,
            box_heights=(box_height_start, box_height_end),
            min_elevation=min_elevation,
            start_direction=start_direction,
            end_direction=end_direction,
            bottom_shift_meter=bottom_shift_meter,
            circular_arch=circular_arch,
            arch_height=arch_height
        )

        arches.append(arch)

    return arches


def create_bridge(
    line3d_swiss,
    deck_width_pair : str ="4.0,4.0",
    bottom_shift_percentage=0,
    min_elevation=0,
    arch_fractions=None,
    pier_size_meters=10,
    circular_arch=True,
    arch_height_fraction=0.8,
):
    logger.debug(f"Create bridge with deck_width={deck_width_pair}, min_elevation={min_elevation:.1f}, circular_arch={circular_arch}, arch_height_fraction={arch_height_fraction}")
    # Convert to list of coordinates if input is a shapely.LineString
    if isinstance(line3d_swiss, LineString):
        logger.debug(f"Received a line segment with {line3d_swiss.length:.1f} meters")
        # Extract coordinates from the LineString
        coords_2d = list(line3d_swiss.coords)
        # We need to ensure we have z-coordinates
        # If the LineString is 2D, we'll need to get z values from elsewhere
        if len(coords_2d[0]) == 2:
            raise ValueError("LineString must have Z coordinates")
        line3d_swiss = coords_2d

    # parse deck_width which could be either a pair of floats or one number
    def parse_deck_width_parameter(input_string: str) -> tuple[float, float]:
        numbers = input_string.strip().split(',')
        if len(numbers) == 1:
            # Single number case: create (a/2, a/2)
            a = float(numbers[0])
            return (a/2, a/2)
        else:
            # Two number case: return (a, b)
            return (float(numbers[0]), float(numbers[1]))
    deck_width_pair = parse_deck_width_parameter(deck_width_pair)
    deck_width = deck_width_pair[0] + deck_width_pair[1]
    
    # Calculate bottom deck width based on the shift percentage
    bottom_deck_width = deck_width + (bottom_shift_percentage * deck_width)

    # Create the bridge deck
    deck_mesh, coords, perp_vectors = create_bridge_deck(
        line_swiss=line3d_swiss,
        deck_width_pair=deck_width_pair,
        min_elevation=min_elevation,
        bottom_shift_percentage=bottom_shift_percentage,
    )

    # how much the piers should move outwards (that means the arch hole gets
    # smaller at the bottom).
    bottom_shift_meter = bottom_shift_percentage * deck_width

    # If no fractions provided, default to a single arch
    if arch_fractions is None:
        arch_fractions = [1.0]

    # Create the arches
    arches = create_arches(
        line_swiss=line3d_swiss,
        deck_width=deck_width,
        min_elevation=min_elevation,
        arch_fractions=arch_fractions,
        arch_height_fraction=arch_height_fraction,
        pier_size_meters=pier_size_meters,
        bottom_shift_meter=bottom_shift_meter,
        circular_arch=circular_arch
    )

    # Add all arches to the scene
    arches_combined = trimesh.util.concatenate(arches)
    bridge_mesh = repair_and_boolean_operation(
        deck_mesh, arches_combined, bool_operation="difference"
    )

    footprint = trimesh.path.polygons.projected(bridge_mesh,
                                                origin=[0,0,min_elevation], normal=[0,0,1],
                                                ignore_sign=True, precise=False)

    return pv.wrap(bridge_mesh), footprint

@click.command()
@click.option(
    "--geojson",
    type=click.Path(exists=False),
    help="Path to GeoJSON file or '-' for STDIN",
)
def main(
    geojson
):
    # Default GeoJSON string (Landwasserviadukt example)
    geojson_str = """
{"features":[{"bbox":[9.675145,46.680561,9.676638,46.681012],"geometry":{"coordinates":[[9.675145,46.680978,1050.852],[9.675196,46.680988,1051.135],[9.675311,46.681003,1051.342],[9.675425,46.681011,1051.542],[9.675538,46.681012,1051.742],[9.675652,46.681005999999996,1051.943],[9.675765,46.680994,1052.143],[9.675875,46.680975,1052.343],[9.675983,46.680949,1052.545],[9.676087,46.680917,1052.745],[9.676186,46.68088,1052.945],[9.676281,46.680836,1053.145],[9.67637,46.680787,1053.345],[9.676452,46.680732,1053.546],[9.676527,46.680673,1053.746],[9.676594,46.68061,1053.946],[9.676638,46.680561,1054.095]],"type":"LineString"},"id":10408,"properties":{"achse_dkm":"Wahr","anschlussgleis":"Falsch","anzahl_spuren":"1","auf_strasse":"Falsch","ausser_betrieb":"Falsch","betriebsbahn":"Falsch","bp_arch_fractions":"0.166,0.166,0.166,0.166,0.166,0.166","bp_arch_height_fraction":0.85,"bp_bottom_shift_percentage":0.4,"bp_circular_arch":true,"bp_deck_width":"5","bp_pier_size":3,"bp_rowid":1,"datum_aenderung":"2023-04-24","datum_erstellung":"2010-07-13","eroeffnungsdatum":null,"erstellung_jahr":2009,"erstellung_monat":8,"grund_aenderung":"Verbessert","herkunft":"swisstopo","herkunft_jahr":2022,"herkunft_monat":6,"id":10408,"kunstbaute":"Bruecke","museumsbahn":"Falsch","name":"Landwasserviadukt","objektart":"Schmalspur","revision_jahr":2022,"revision_monat":6,"revision_qualitaet":"Akt","standseilbahn":"Falsch","stufe":"1","tlm_oev_name_uuid":"{20E2723B-F0FF-462B-9FAA-556F26185936}","uuid":"{DBDFB80A-837C-41B4-8E27-871C4C2CCC64}","verkehrsmittel":"Bahn","zahnradbahn":"Falsch"},"type":"Feature"}],"type":"FeatureCollection"}
    """

    # Load GeoJSON data from file, STDIN, or use default
    if geojson:
        if geojson == '-':
            # Read from STDIN
            import sys
            geojson_str = sys.stdin.read()
            logger.info("Reading GeoJSON from STDIN")
        else:
            # Read from file
            with open(geojson, 'r') as f:
                geojson_str = f.read()
            logger.info(f"Reading GeoJSON from file: {geojson}")
    else:
        logger.info("Using default GeoJSON (Landwasserviadukt)")

    try:
        data = json.loads(geojson_str)
    except json.decoder.JSONDecodeError:
        logger.critical(f"Are you sure we have a GeoJSON here: '{geojson_str[0:25]} ...'?")
        sys.exit(-1)


    match_zero_pair_pattern = r'^\s*(0(?:\.0+)?\s*,\s*0(?:\.0+)?|0(?:\.0+)?)\s*$'
     
    all_meshes=[]
    for i, feature in enumerate(data["features"]):
        # Check if bridge parameters are in the GeoJSON properties
        properties = feature.get("properties", {})

        uuid = properties.get("uuid", "Unknown UUID")
        logger.info(f"Feature {i} ({uuid})")

        coords = feature["geometry"]["coordinates"]

        # Set up coordinate transformation from WGS84 to Swiss CH1903+ (EPSG:2056)
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:2056", always_xy=True)

        # Convert to Swiss coordinates (EPSG:2056)
        swiss_coords = []
        for point in coords:
            lon, lat, elev = point
            x, y = transformer.transform(lon, lat)
            swiss_coords.append([x, y, elev])
                
        # Create a 3D LineString from the points
        line3d_swiss = LineString(
            [(p[0], p[1], p[2]) for p in swiss_coords]
        )  # 2D line for shapely operations


        anzahl_spuren = properties.get("anzahl_spuren", None)
        objektart = properties.get("objektart", None)

        logger.info(f"- anzahl_spuren={anzahl_spuren}, objektart={objektart}")
        params_tuple = define_bridge_parameters(line3d_swiss.length,
                                                anzahl_spuren=anzahl_spuren,
                                                objektart=objektart)
        deck_width, bottom_shift_percentage, arch_fractions, pier_size_meters, arch_height_fraction, circular_arch = params_tuple

        
        max_z = max(point[2] for p in swiss_coords)
        min_elevation = get_min_height_swissalti_service(line3d_swiss) - 10
        bridge_height = max_z - min_elevation
        logger.info(f"- I found a min elevation of {min_elevation:.2f} [m], bridge height is {bridge_height:.2f} [m]")

        # Override command line parameters with GeoJSON properties if they exist
        if "bp_deck_width" in properties and properties.get("bp_deck_width", None) is not None:
            deck_width = properties["bp_deck_width"]
            logger.info(f"- Using deck width from GeoJSON: {deck_width}")

        if "bp_bottom_shift_percentage" in properties and properties.get("bp_bottom_shift_percentage", None) is not None:
            bottom_shift_percentage = float(properties["bp_bottom_shift_percentage"])
            logger.info(f"- Using bottom shift percentage from GeoJSON: {bottom_shift_percentage}")

        if "bp_arch_height_fraction" in properties and properties.get("bp_arch_height_fraction", None) is not None:
            arch_height_fraction = float(properties["bp_arch_height_fraction"])
            logger.info(f"- Using arch height fraction from GeoJSON: {arch_height_fraction}")

        if "bp_pier_size" in properties and properties.get("bp_pier_size", None) is not None:
            pier_size = float(properties["bp_pier_size"])
            logger.info(f"- Using pier size from GeoJSON: {pier_size}")

        if "bp_circular_arch" in properties and properties.get("bp_circular_arch", None) is not None:
            circular_arch = bool(properties["bp_circular_arch"])
            logger.info(f"- Using circular arch setting from GeoJSON: {circular_arch}")

        if "bp_arch_fractions" in properties and properties.get("bp_arch_fractions", None) is not None:
            arch_fractions = properties["bp_arch_fractions"]
            logger.info(f"- Using arch fractions from GeoJSON: {arch_fractions}")
        

        logger.info(f"- Input Parameters: shapely_geom={line3d_swiss} (length = {line3d_swiss.length:.2f}, deck_width_pair={deck_width}, "
                    f"bottom_shift_percentage={bottom_shift_percentage}, min_elevation={min_elevation}, "
                    f"arch_fractions={arch_fractions}, pier_size_meters={pier_size_meters}, "
                    f"circular_arch={circular_arch}, arch_height_fraction={arch_height_fraction}")
        if not re.match(match_zero_pair_pattern, deck_width):
            bridge_mesh, footprint = create_bridge(
                line3d_swiss,
                deck_width_pair=deck_width,
                bottom_shift_percentage=bottom_shift_percentage,
                min_elevation=min_elevation,
                arch_fractions=arch_fractions,
                pier_size_meters=pier_size_meters,
                circular_arch=circular_arch,
                arch_height_fraction=arch_height_fraction,
            )
            all_meshes.append(bridge_mesh)
        else:
            logger.warning(f"- I skip this bridge as deck_width is set to 0")
        logger.info("Done")
        
    if len(all_meshes):
        plotter = pv.Plotter()
        for mesh in all_meshes:
            plotter.add_mesh(mesh)
        plotter.show()
    else:
        logger.critical("No bridges found to display")



if __name__ == "__main__":
    # python scripts/bridge_creation.py --no-circular-arch --arch-fractions "0.166,0.166,0.166,0.166,0.166,0.166" --pier-size 5 --bottom-shift-percentage 0.4 --bridge-height 65 --arch-height-fraction 0.8 --circular-arch
    main()
