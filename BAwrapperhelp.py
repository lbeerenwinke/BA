import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LinearRing, LineString
from shapely.affinity import translate
import BAhelp as help
from BAhelp import compute_segments_from_points
from matplotlib.ticker import ScalarFormatter
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import os




# ----------------------------------------------------------------------------------------------
# ATTETION POINTS VS SEGMENTS, SOME FUNCTION CALCULATE ONE, SOME THE OTHER - ALL RETURN SEGMENTS
# ----------------------------------------------------------------------------------------------

def generate_ellipse(a, b, num_points):
    """
    Generate points on an ellipse with semi-major axis `a` and semi-minor axis `b`.
        a (float): horizontal radius
        # b (float): vertical radius
    """
    # Create the angles for the parametric equation (from 0 to 2*pi)
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # endpoint=False avoids duplicate point at 2*pi
    
    # Parametric equations for ellipse (counterclockwise by default)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Combine x and y into a single array of points
    points = np.column_stack((x, y))

    # Reverse the points to ensure clockwise order
    points = points[::-1]

    return compute_segments_from_points(points)



def generate_rectangle(width, height):
    """Generates a clockwise rectangle with width `width` and height `height`."""

    points = np.array([
        [0, 0],            # Bottom-left
        [0, height],       # Bottom-right
        [width, height],   # Top-right
        [width, 0]        # Top-left
    ])
    
    return compute_segments_from_points(points)

def generate_triangle(base, height):
    """Generates a clockwise triangle with a given base and height."""
	
    points = np.array([
        [0, 0],             # Bottom-left
        [base / 2, height],  # Top
        [base, 0]          # Bottom-right
    ])
    
    return compute_segments_from_points(points)

def generate_star(arms, radius_outer, radius_inner, num_points=None):
    """
    Generates a star shape with `arms`, `outer radius`, and `inner radius`.
        arms (int): Number of arms of the star.
        radius_outer (float): Radius of the outer points of the star.
        radius_inner (float): Radius of the inner points of the star.
    """
    if num_points is None:
        num_points = 2 * arms
    
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    radius = np.empty(num_points)
    radius[::2] = radius_outer  # Outer points
    radius[1::2] = radius_inner  # Inner points
    
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    
    points = np.column_stack((x, y))
    
    return compute_segments_from_points(points[::-1])  # Reverse to ensure clockwise


# "Moon/paraglider" shape
def generate_ellipse_above_diff(param, a=5, b=2, c=5, d=2, num_points=100, gap=2.5):
    """
    Generates the difference between two ellipses where the second one is gradually
    moved up to intersect the first.
        param (float): A value between 0 and 1 indicating how much the ellipses overlap.
        a (float): analogue ellipse
        b (float): analogue ellipse
        gap (float): Initial vertical gap between the ellipses when param=0.
    """
    
    # Generate base ellipses
    ellipse_a = generate_ellipse_points_helper(a, b, num_points)
    ellipse_b = generate_ellipse_points_helper(c, d, num_points)

    poly_a = Polygon(ellipse_a)
    
    # Move ellipse B based on parameter
    shift = - (b + d) - gap + ((b + d) + gap) * param
    ellipse_b_shifted = translate(Polygon(ellipse_b), yoff=shift)

    # Compute the difference
    result = poly_a.difference(ellipse_b_shifted)


    # No segment conversion here — we work directly with points
    ellipse_a_pts = ellipse_a  # Already raw points
    ellipse_b_pts = np.array(ellipse_b_shifted.exterior.coords[:-1])  # Drop duplicate closing point

#     # Visualize both before the difference operation
#     help.plot_shape_multi_from_points(
#     ellipse_a_pts,
#     ellipse_b_pts,
#     labels=[f'A (Top Ellipse)', f'B (Shifted Bottom Ellipse)'],
#     title="Original and Shifted Ellipses"
# )

    if result.is_empty:
        return compute_segments_from_points(np.empty((0, 2)))
    

    def get_cw_coords(polygon):
        coords = np.array(polygon.exterior.coords)
        coords = coords[:-1]  # remove the duplicate end point
        ring = LinearRing(coords)
        if ring.is_ccw:
            coords = coords[::-1]
        return coords

    # Handle Polygon or MultiPolygon
    if result.geom_type == 'Polygon':
        return compute_segments_from_points(get_cw_coords(result))
    elif result.geom_type == 'MultiPolygon':
        largest = max(result.geoms, key=lambda g: g.area)
        return compute_segments_from_points(get_cw_coords(largest))
    return compute_segments_from_points(np.empty((0, 2)))



# "Moon/paraglider" shape
def generate_ellipse_above_diff_flat_base(param, a=5, b=2, c=5, d=0, num_points=100):
    """
    Generates the difference between two ellipses where the second one is gradually
    moved up to intersect the first.
        param (float): A value between 0 and 1 indicating how much the ellipses overlap.
        a (float): analogue ellipse
        b (float): analogue ellipse
        gap (float): Initial vertical gap between the ellipses when param=0.
    """
    
    # Generate base ellipses
    ellipse_a = generate_upper_ellipse_with_flat_base(a, b, num_points)
    # Linearly interpolate d → b
    d_param = d + (b - d) * param
    ellipse_b = generate_ellipse_points_helper(c, d_param, num_points)

    poly_a = Polygon(ellipse_a)
    
    # Move ellipse B based on parameter
    ellipse_b_shifted = translate(Polygon(ellipse_b), yoff=0)

    # Compute the difference
    result = poly_a.difference(ellipse_b_shifted)


    # No segment conversion here — we work directly with points
    ellipse_a_pts = ellipse_a  # Already raw points
    ellipse_b_pts = np.array(ellipse_b_shifted.exterior.coords[:-1])  # Drop duplicate closing point


    # # Visualize both before the difference operation
    # help.plot_shape_multi_from_points(
    #     ellipse_a_pts,
    #     ellipse_b_pts,
    #     labels=[f'A (Top Ellipse)", "B (Shifted Bottom Ellipse) {param}'],
    #     title="Original and Shifted Ellipses"
    # )


    if result.is_empty:
        return compute_segments_from_points(np.empty((0, 2)))
    

    def get_cw_coords(polygon):
        coords = np.array(polygon.exterior.coords)
        coords = coords[:-1]  # remove the duplicate end point
        ring = LinearRing(coords)
        if ring.is_ccw:
            coords = coords[::-1]
        return coords

    # Handle Polygon or MultiPolygon
    if result.geom_type == 'Polygon':
        return compute_segments_from_points(get_cw_coords(result))
    elif result.geom_type == 'MultiPolygon':
        largest = max(result.geoms, key=lambda g: g.area)
        return compute_segments_from_points(get_cw_coords(largest))
    return compute_segments_from_points(np.empty((0, 2)))



def generate_scaled_ellipse_diff(param, a=5, b=2, c=5, d=2, num_points=100, gap=2.5, target_length=50.0):
    """
    Generates a scaled difference shape of two ellipses with approximately equal length.
    
    Parameters:
        param (float): overlap parameter
        target_length (float): desired total length of the resulting shape
        tolerance (float): acceptable deviation in length
    """
    original_segments = generate_ellipse_above_diff(param, a, b, c, d, num_points, gap)

    def compute_length(segments):
        return np.sum(np.linalg.norm(segment) for segment in segments)

    original_length = compute_length(original_segments)

    scale_factor = np.array(target_length / original_length)

    # Scale all segments
    scaled_segments = [(seg * scale_factor) for seg in original_segments]
    return scaled_segments





def generate_naca_0012_truncated(scale=1.0):
    """Returns scaled NACA 0012 truncated profile."""
    coords = np.array([
        (0.45, 0.72), (1.30, 0.41), (1.99, 0.21),
        (2.44, -0.02), (2.60, -0.10), (2.46, -0.13),
        (2.08, -0.22), (1.53, -0.26), (0.92, -0.28),
        (0.38, -0.33), (-0.38, -0.33), (-0.92, -0.28),
        (-1.53, -0.26), (-2.08, -0.22), (-2.46, -0.13),
        (-2.60, -0.10), (-2.44, -0.02), (-1.99, 0.21),
        (-1.30, 0.41), (-0.45, 0.72)
    ])
    return coords * scale

def generate_symmetrical_umbrella(scale=1.0):
    """Symmetrical umbrella shape with scaling."""
    coords = np.array([
        (2, 2), (4, 0), (2, -2), (-2, 2), (-4, 0), (-2, -2)
    ])
    return coords * scale

def generate_asymmetrical_umbrella(scale=1.0):
    """Asymmetrical umbrella shape with longer right side."""
    coords = np.array([
        (2, 2), (4, 0), (3, -3), (-3, 3), (-4, 0), (-2, -2)
    ])
    return coords * scale

def generate_alternate_umbrella(scale=1.0):
    """Alternative umbrella with sharper angles."""
    coords = np.array([
        (10, 10), (20, 0), (20, -20), (-20, 20), (-20, 0), (-10, -10)
    ])
    return coords * scale

def generate_paraglider(scale=1.0):
    """ Simple paraglider shape."""
    coords = np.array([
        [0.5, 0.5], [1, 0.5], [3, 0], [1, -0.5], [0.5, -0.5], [0, -0.5],
        [-0.5, 0.5], [-1, 0.5], [-3, 0], [-1, -0.5], [-0.5, -0.5], [0, 0.5]
    ])
    return coords * scale



def generate_ellipse_points_helper(a, b, num_points):
    """
    Generate points on an ellipse with semi-major axis `a` and semi-minor axis `b`.
        a (float): horizontal radius
        b (float): vertical radius
    """
    # Create the angles for the parametric equation (from 0 to 2*pi)
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # endpoint=False avoids duplicate point at 2*pi
    
    # Parametric equations for ellipse (counterclockwise by default)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Combine x and y into a single array of points
    points = np.column_stack((x, y))

    # Reverse the points to ensure clockwise order
    points = points[::-1]

    return points


def generate_upper_ellipse_with_flat_base(a, b, num_points):
    """
    Generate points on the upper half of an ellipse with a flat base.
    
    Parameters:
        a (float): Horizontal radius (semi-major axis)
        b (float): Vertical radius (semi-minor axis)
        num_points (int): Number of points to generate (including the flat base)
    
    Returns:
        segments: Segments computed from the modified shape
    """
    # Split the number of points roughly between arc and base
    num_arc_points = num_points // 2
    num_line_points = num_points - num_arc_points

    # Generate angles for the upper half (0 to pi)
    t = np.linspace(0, np.pi, num_arc_points, endpoint=False)
    t = t[1:]
    t = t[::-1]
    
    # Upper arc of the ellipse
    x_arc = a * np.cos(t)
    y_arc = b * np.sin(t)

    # Flat base (horizontal line connecting the ends)
    x_line = np.linspace(-a, a, num_line_points)
    x_line = x_line[::-1]
    y_line = np.zeros_like(x_line)
    y_line = y_line[::-1]

    # Combine arc and line
    x = np.concatenate((x_arc, x_line))
    y = np.concatenate((y_arc, y_line))

    points = np.column_stack((x, y))

    return points

