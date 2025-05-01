import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import math
from shapely.geometry import LineString
from matplotlib.ticker import ScalarFormatter


# Physical constant (air density)
pf = 1.2041

# -------------------------------
# INITIALIZATION HELPERS
# -------------------------------
def initialize_points(start_x, start_y, inp):
    """Create absolute points from relative segments starting at (start_x, start_y)."""
    points = [(0, 0)] * len(inp)
    points[0] = (start_x, start_y)
    for i in range(len(inp) - 1):
        points[i + 1] = (points[i][0] + inp[i][0], points[i][1] + inp[i][1])
    return np.array(points)

def make_closed_segments(segments):
    """Append inverted segments in reverse order to close the (line) shape."""
    inverted = [(-dx, -dy) for dx, dy in reversed(segments)]
    return segments + inverted

def discretization(segments, divisions):
    """Divide each segment into smaller subsegments, based on the given division factor."""
    refined = []
    for dx, dy in segments:
        for _ in range(divisions):
            refined.append((dx / divisions, dy / divisions))
    return refined

# -------------------------------
# CENTER OF MASS
# -------------------------------
def compute_com_from_points(points):
    """Compute the center of mass (COM) from a list of points."""

    total_length = 0  # Sum of all segment lengths
    weighted_x = 0  # Weighted sum of x-coordinates
    weighted_y = 0  # Weighted sum of y-coordinates


# Loop through consecutive pairs of points to compute segment lengths and midpoints
    for i in range(len(points)):
        x1, y1 = points[i % len(points)]
        x2, y2 = points[(i + 1) % len(points)]
        
        # Compute segment length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Compute midpoint of the segment
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        
        # Weight the midpoint by the segment length
        weighted_x += midpoint_x * length
        weighted_y += midpoint_y * length
        
        # Sum total length
        total_length += length

    return (weighted_x / total_length, weighted_y / total_length)

def compute_com_from_points_with_mass(points, com_shifted, new_mass, mass_scale):
    """Extended COM calculation including an external point mass."""

    # Compute the original COM from the points
    total_length = 0  # Sum of all segment lengths
    weighted_x = 0  # Weighted sum of x-coordinates
    weighted_y = 0  # Weighted sum of y-coordinates

    # Loop through consecutive pairs of points to compute segment lengths and midpoints
    for i in range(len(points)):
        x1, y1 = points[i % len(points)]
        x2, y2 = points[(i + 1) % len(points)]
        
        # Compute segment length
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2) * mass_scale
        
        # Compute midpoint of the segment
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        
        # Weight the midpoint by the segment length
        weighted_x += midpoint_x * length
        weighted_y += midpoint_y * length
        
        # Sum total length
        total_length += length

    # Weighted sum for the new COM
    weighted_x += com_shifted[0] * new_mass
    weighted_y += com_shifted[1] * new_mass
    # Calculate the new COM considering the new point and its mass
    # The mass of the system includes the original length-based mass and the new mass
    total_mass = total_length + new_mass
    return (weighted_x / total_mass, weighted_y / total_mass)


def compute_vectors_to_com(points, com):
    """Compute vectors from each point to the center of mass (COM)."""
    vectors = [(com[0] - x, com[1] - y) for x, y in points]
    return vectors


# -------------------------------
# FORCE CALCULATIONS, GRAVITY AND COMBINED AERODYNAMIC FORCE
# -------------------------------
def compute_fg(segment, ag, mass_scale):
    """Gravity force on a segment."""
    return (ag[0], ag[1] * scaled_segment_length(segment, mass_scale))



def compute_force(segment, startpoint, v_object, ang_vel, com):
    """Combinining equation for aerodynamic force (lift and drag) acting on a segment."""
    # Segment length
    segment_length = np.linalg.norm(segment)
    
    if (segment_length == 0):
        return (0, 0)  # Avoid division by zero
    
    # Compute unit normal vector, CCW
    n = (-segment[1] / segment_length, segment[0] / segment_length)

    # Midpoint of the segment
    midpoint = (startpoint[0] + segment[0] / 2, startpoint[1] + segment[1] / 2)

    # Tangential velocity contribution due to angular velocity
    r = (midpoint[0] - com[0], midpoint[1] - com[1])  # Vector from COM to midpoint
    ang_3d = (0, 0, ang_vel)
    tangential_velocity = np.cross(ang_3d, r)  # Cross product w x r

    # Total velocity at the midpoint (linear + tangential)
    v_p = (v_object[0] + tangential_velocity[0], v_object[1] + tangential_velocity[1])

    # Force coefficient
    v_p_magnitude = np.linalg.norm(v_p)
    dot_product = np.dot(v_p, n)
    coeff = -pf * segment_length * v_p_magnitude * dot_product

    # Aerodynamic force
    force = (coeff * n[0], coeff * n[1])

    return force


# -------------------------------
# AERODYNAMIC FORCE HELPERS
# -------------------------------

def scaled_segment_length(segment, scale):
    return scale * np.linalg.norm(segment)

def is_segment_blocked(segment, startpoint, v_p, all_points, segment_index, ray_length=100, special_index=2000):
    """Check if a segment is blocked from airflow by others."""
    # Compute the midpoint of the segment
    midpoint = (startpoint[0] + segment[0] / 2, startpoint[1] + segment[1] / 2)

    v_p_norm = np.linalg.norm(v_p)
    if(v_p_norm == 0):
        return True
    unit_v_object_vector = (v_p[0]/v_p_norm, v_p[1]/v_p_norm)

    # Define the ray starting at midpoint in the v_object direction
    ray_end = (
        midpoint[0] + unit_v_object_vector[0] * ray_length,
        midpoint[1] + unit_v_object_vector[1] * ray_length
    )
    airflow_ray = LineString([midpoint, ray_end])

    # Check for intersections with other segments
    for i in range(len(all_points)):
        if i == segment_index or i == special_index:
            continue  # Skip self
        seg_line = LineString([all_points[i % len(all_points)], all_points[(i + 1) % len(all_points)]])
        if airflow_ray.intersects(seg_line):
            return True  # It's blocked

    return False  # It's not blocked



def compute_deflected_airflow_force(alpha, v_object, ang_vel, segment, pt, 
                                 all_segments, all_points,
                                 com,
                                 segment_index,
                                 neighbour_level):
    """Recursive computation of deflected aerodynamic forces."""
    # Midpoint of the deflecting segment (origin of deflected airflow)
    midpoint = np.array((pt[0] + segment[0] / 2, pt[1] + segment[1] / 2))

    # Tangential velocity contribution due to angular velocity
    r = (midpoint[0] - com[0], midpoint[1] - com[1])  # Vector from COM to midpoint
    ang_3d = (0, 0, ang_vel)
    tangential_velocity = np.cross(ang_3d, r)  # Cross product w x r

    # Total velocity at the midpoint (linear + tangential)
    v_p = np.array((v_object[0] + tangential_velocity[0], v_object[1] + tangential_velocity[1]))

    # Compute segment normal (CCW)
    norm = np.linalg.norm(segment)
    if norm == 0:
        return np.zeros(2)

    normal = np.array([-segment[1], segment[0]]) / norm

    # Reflect airflow velocity at the segment's normal
    v_airflow = -v_p
    deflected_v_airflow = v_airflow - 2 * np.dot(v_airflow, normal) * normal
    deflected_v_p = -deflected_v_airflow  # for force calculation

    total_force = np.zeros(2)

    for j, (seg_target, pt_target) in enumerate(zip(all_segments, all_points)):
        if j == segment_index:
            continue  # Skip self

        seg_target = np.array(seg_target)
        pt_target = np.array(pt_target)

        # Compute midpoint of target segment
        target_mid = np.array((pt_target[0] + seg_target[0] / 2, pt_target[1] + seg_target[1] / 2))

        # Check if segment is far away
        if (np.linalg.norm(segment_index - j) >= neighbour_level):
            continue

        # Check if target segment lies in general direction of deflected airflow
        
        if not is_segment_blocked(segment=segment, startpoint=pt, v_p=deflected_v_airflow, 
                                all_points=[pt_target, pt_target + seg_target], segment_index=3): # Index three to avoid it, anything except 0 works
            continue  # Not in direction

        # Check for occlusion (ray from origin to target midpoint blocked by any segment)
        if is_segment_blocked(segment=seg_target, startpoint=pt_target, v_p=deflected_v_p,
                                all_points=all_points, segment_index=j, ray_length= (np.linalg.norm(midpoint-target_mid)), special_index=segment_index):
            continue  # Blocked by another segment

        # Passed both checks: compute force
        force = (1 / np.linalg.norm(midpoint - target_mid)) * alpha * np.array(
            compute_force(seg_target, pt_target, deflected_v_p, ang_vel, com)
        )
        total_force += force

    return total_force




# -------------------------------
# INERTIA AND TORQUE
# -------------------------------



def compute_moment_of_inertia_from_points(points, com):
    """Compute the total moment of inertia for a line defined by points,
    where the mass of each segment is proportional to its length."""

    I_total = 0  # Initialize total moment of inertia
    
    # Iterate over consecutive points to define segments
    for i in range(len(points)):
        x1, y1 = points[i % len(points)]
        x2, y2 = points[(i + 1) % len(points)]
        
        # Compute segment length (mass is proportional to length)
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        mass_segment = length  # Mass is equal to the length
        
        # Compute segment center of mass (midpoint)
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        
        # Compute distance from segment midpoint to system COM
        dx = midpoint_x - com[0]
        dy = midpoint_y - com[1]
        distance_squared = dx**2 + dy**2

        # Moment of inertia for the segment
        I_center = (1/12) * 0.1 * length**3  # About segment's center, length = mass
        I_segment_com = I_center + mass_segment * distance_squared  # Parallel axis theorem

        # Add to total moment of inertia
        I_total += I_segment_com

    return I_total

def compute_moment_of_inertia_from_manual_pos(manual_pos, manual_mass, com):
    """Compute the moment of inertia for the single manual point mass."""
    # Compute the distance from the manual_pos to the center of mass (COM)
    dx = manual_pos[0] - com[0]
    dy = manual_pos[1] - com[1]
    
    # Compute the distance squared (r^2)
    r_squared = dx**2 + dy**2
    
    # Moment of inertia for a point mass
    I_manual_pos = manual_mass * r_squared
    
    return I_manual_pos

def compute_torque(segment, startpoint, currfres, com):
    """Compute torque from the accumulated force on a segment."""

    # Midpoint and radius vector
    midpoint = (startpoint[0] + (segment[0]/2), startpoint[1] + (segment[1]/2))
    r = (midpoint[0] - com[0], midpoint[1] - com[1])

    # Torque
    torque = np.cross(r, currfres)

    return torque

def compute_torque_manual_pos(pos, currfres, com):
    """Compute torque caused by force on the manual mass."""
    
    # Radius vector
    r = pos - com

    # Torque
    torque = np.cross(r, currfres)
    

    return torque


# -------------------------------
# UPDATE HELPERS
# -------------------------------


def rotate_points_around_com_degrees(points, angle, com, in_degrees):
    """Rotate a list of points around the COM by a specified angle in degrees or radians"""
    if (angle == 0):
        return (points)
    if in_degrees:
        angle = math.radians(angle)  # Convert degrees to radians
    return rotate_points_around_com(points, angle, com)
    


def rotate_points_around_com(points, angle, com):
    """Rotate a list of points around the COM by a specified angle in radians."""
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Shift points to COM
    shifted_points = [(x - com[0], y - com[1]) for x, y in points]
    
    # Apply rotation and shift back to COM
    rotated_points = []
    for x, y in shifted_points:
        rotated = np.dot(rotation_matrix, np.array([x, y]))  # Rotate the vector
        new_x = rotated[0] + com[0]
        new_y = rotated[1] + com[1]
        rotated_points.append((new_x, new_y))
    
    return rotated_points

def compute_segments_from_points(points):
    """Compute segments or difference vectors (directional vectors) from a list of points."""
    
    segments = []

    # Loop through consecutive pairs of points to compute difference vectors
    for i in range(len(points)):
        dx = points[(i + 1) % len(points)][0] - points[i % len(points)][0]  # Difference in x
        dy = points[(i + 1) % len(points)][1] - points[i % len(points)][1]  # Difference in y
        segments.append((dx, dy))
    
    return segments




# -------------------------------
# PLOTTING AND IT'S HELPERS
# -------------------------------

def add_first(points):
    """Appends the first point to the end of the list to (visually) close the loop."""

    points = np.asarray(points)
    return np.vstack([points, points[0]])


def plot_shape_deflected_airflow(segments, title="Shape", save=False, save_path=r"C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\deflected_airflow_undef.png"):
    """
    Plot a shape from segments and show two aerodynamic force vectors (airflow from below).
    """
    
    # Init shape
    points = initialize_points(0, 0, segments)
    points = add_first(points)
    segments = compute_segments_from_points(points)

    x, y = points[:, 0], points[:, 1]

    # Figure and axis setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, 'o-', label="Shape outline")
    ax.set_aspect('equal', adjustable='box')

    # Padding
    x_margin = (x.max() - x.min()) * 0.2 or 1.0
    y_margin = (y.max() - (-3)) * 0.1 or 1.0
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim((-3) - y_margin, y.max() + y_margin)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()

    # ---- Add aerodynamic force vectors ----
    # Assumptions
    v_object = np.array([0, -1])  # meaning airflow from below
    ang_vel = 0.0                # No rotation
    com = compute_com_from_points(points)  # Center of mass

    # Choose two segments for demo
    selected_indices = [2, 3] if len(points) > 4 else [0]

    for i in selected_indices:
        segment = segments[i]
        startpoint = points[i]
        midpoint = startpoint + 0.5 * np.array(segment)
        force = compute_force(segment, startpoint, v_object, ang_vel, com)
        ax.quiver(midpoint[0], midpoint[1], force[0], force[1],
                  angles='xy', scale_units='xy', scale=1, color='red', label='Aerodynamic force' if i == selected_indices[0] else None)

    # ---- Save or show plot ----
    if save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")

    plt.show()



def plot_shape(segments, title="Shape",  plot_show = True, save=True, save_path="C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\plotshape.png"):
    """Plots a list of segments as a 2D shape."""
    
    points = initialize_points(0, 0, segments)
    points = add_first(points)
    x, y = points[:, 0], points[:, 1]

    # Explicitly create and keep track of figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, 'o-', label="Shape outline")
    ax.set_aspect('equal', adjustable='box')

    # Padding and limits
    x_margin = (x.max() - x.min()) * 0.1 or 1.0
    y_margin = (y.max() - y.min()) * 0.1 or 1.0
    ax.set_xlim(x.min() - x_margin, x.max() + x_margin)
    ax.set_ylim(y.min() - y_margin, y.max() + y_margin)

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.legend()

    if save:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved as '{save_path}'")
    
    if plot_show:
        plt.show()



def plot_shape_multi_from_points(*point_sets, labels=None, title="Multiple Shapes (Points)", plot_show = True, save=True, save_path="C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\plotshapemultiplefrompoints.png"):
    """
    Plots multiple shapes given as lists of 2D points (not segments).
    
    Parameters:
        *point_sets: variable number of np.array point lists (each N x 2).
        labels: list of labels for the shapes.
        title: plot title.
        save: whether to save the plot.
        save_path: path to save the plot (if save=True).
    """
    plt.figure(figsize=(6, 6))
    
    for i, points in enumerate(point_sets):
        points = np.array(points)
        if points.shape[0] > 0:
            points = np.vstack([points, points[0]])  # Close the loop
            label = labels[i] if labels and i < len(labels) else f"Shape {i+1}"
            plt.plot(points[:, 0], points[:, 1], 'o-', label=label)
    
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    
    if save and save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f" Plot saved to '{save_path}'")
    
    if plot_show:
        plt.show()


def plot_simulation_scores(all_scores, show=[True, True, True], plot_show = True, title="Simulation Score Comparison", save=False, save_path = None):
    """
    Plots translational, rotational, and height scores for multiple shapes using a color gradient.
    Rotational score is plotted on a secondary Y-axis (right).

    Parameters:
    - all_scores: list of dicts with keys: 'name', 'time', 'translational', 'rotational', 'height'
    - show: list of 3 booleans [show_translational, show_rotational, show_height]
    - title: title for the plot
    - save: if True, saves the plot
    - save_path: file path to save the figure
    """
    # Define consistent styles
    score_styles = {
        'translational': '-',
        'rotational': '--',
        'height': ':'
    }

    # Use a gradient colormap instead of fixed colors
    cmap = plt.cm.plasma_r

    num_scores = len(all_scores)
    colors = [cmap(i / max(num_scores - 1, 1)) for i in range(num_scores)]

    plt.close('all')

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()  # Right y-axis for rotational

    # Disable scientific notation
    ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax1.ticklabel_format(style='plain', axis='y')
    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax2.ticklabel_format(style='plain', axis='y')

    for idx, scores in enumerate(all_scores):
        color = colors[idx]
        name = scores['name']
        time = scores['time']

        if show[0]:
            ax1.plot(time, scores['translational'], linestyle=score_styles['translational'],
                     color=color, label=f"{name} (Translational)")
        if show[1]:
            ax2.plot(time, scores['rotational'], linestyle=score_styles['rotational'],
                     color=color, label=f"{name} (Rotational)")
        if show[2]:
            ax1.plot(time, scores['height'], linestyle=score_styles['height'],
                     color=color, label=f"{name} (Height)")
        

    ax1.set_title(title)
    ax1.set_xlabel("Time (s)")
    if show[0] or show[2]:
        ax1.set_ylabel("Translational Velocity")
    if show[1]:
        ax2.set_ylabel("Angular Velocity")

    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.tight_layout()

    if save and save_path is None:
        filename = f"{title.replace(' ', '_')}.png"
        save_path = fr"C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\{filename}"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Simulation scores saved to '{save_path}'")

    if plot_show:
        plt.show()






















# -----------------------------------------------------------------------
# FROM HERE ON DOWNWARD ONLY CODE THAT IS NOT USED ANYMORE (BUT ONCE WAS)
# -----------------------------------------------------------------------



# -------------------------------
# PLOTTING 
# -------------------------------




def plot_lines(timestamps, points_per_timestamp, com_values):
    """Plots lines connecting points for each timestamp."""
    
    
    # Set up the plot
    plt.figure(figsize=(8, 6))
    
    # Loop through each timestamp to plot the corresponding lines
    for i, points in enumerate(points_per_timestamp):
        x_values = [p[0] for p in points]  # Get all x coordinates for this timestamp
        y_values = [p[1] for p in points]  # Get all y coordinates for this timestamp
        
        # Plot the line connecting the points
        plt.plot(x_values, y_values, label=f'Timestamp {timestamps[i]}', marker='o')

        # Plot the COM as a disconnected point with the same color
        com_x, com_y = com_values[i]
        plt.scatter(com_x, com_y, color=plt.gca().lines[-1].get_color(), marker='x', s=100, label=f'COM {timestamps[i]}')

    # Ensure equal scaling for both axes
    plt.axis("equal")  # This makes the x and y axes equal in scale

    # Set plot details
    plt.title('Line Simulation Over Time')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    
    # Display the plot
    plt.show()

def animate_lines_sequentially(lines, interval=10, **kwargs):
    """Animate lines sequentially, displaying one after the other with dynamic scaling."""
    

    # Flatten all points to calculate global bounds
    
    all_numbers = [coord for line in lines for point in line for coord in point]

    x_min, x_max = min(all_numbers), max(all_numbers)
    y_min, y_max = min(all_numbers), max(all_numbers)
    
    # Add padding to make the plot visually comfortable
    x_padding = (x_max - x_min) * 0.1 if (x_max - x_min) > 0 else 1.0
    y_padding = (y_max - y_min) * 0.1 if (y_max - y_min) > 0 else 1.0

    fig, ax = plt.subplots()
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.set_aspect('equal', adjustable='box')  # Ensure equal scaling for both axes

    # Line object that will be updated in the animation
    line_plot, = ax.plot([], [], **kwargs)

    def update(frame):
        """Update function to draw the current line."""
        if frame < len(lines):
            current_line = lines[frame]
            line_plot.set_data(
                [point[0] for point in current_line],
                [point[1] for point in current_line]
            )
        return line_plot,

    ani = FuncAnimation(
        fig,
        update,
        frames=len(lines),  # Number of frames equals the number of lines
        interval=interval,
        blit=True
    )
    
    plt.show()

