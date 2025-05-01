import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import BAhelp as help
from matplotlib.ticker import ScalarFormatter

# currpoints = None
# currsegs = None
# currv = None
# curr_ang_v = 0
# iteration = 0
# translational_velocities = []
# angular_velocities = []
# height_loss = []
# com_shift = None
# manual_pos = None


def run_simulation(inp_in, com_shift_in=np.array([0, -10]), initial_angle_in=0, initial_velocity =  np.array([0.0, 0.0]), initial_ang_velocity = 0.0,
                   is_manual_mass=True, manual_mass=60, mass_scale = 0.1, flags = [True, True, True, False], deflected_airflow = False, num_discretization=1, 
                   max_iteration= 101, dtime=0.1, alpha = 0,
                   gravity_acceleration=np.array([0.0, -9.81]), plot_show = True):
    """
    Parameters:
        inp: Input segments for the object (e.g., shape of the object)
        com_shift_in: Defines the position of the manual mass
        initiail_angle_in: The amount the object gets rotated initially
        is_manual_mass: Whether to use a manual mass
        manual_mass: Manual mass value
        flags: Which vectors should be shown in the plot (gravity of manual mass, gravity general, force of the aero, normals of the forces)
        deflected_airflow: Boolean if deflected airflow is turned on, (turned off for faster animation, but "wrong")
        num_discretization: The number of subdivisions for each segment
        max_iteration: Maximum number of iterations in the simulation
        dtime: Time step for the simulation
        gravity_acceleration: Gravitational acceleration vector
    """
    global currpoints, currsegs, currv, curr_ang_v, iteration, translational_velocities, angular_velocities, com_shift, manual_pos
    
    # Truncate NumPy arrays to 3 decimal places
    np.set_printoptions(precision=5, suppress=True)

    # -----------------------------
    # Simulation setup
    # -----------------------------

    # FLAGS to control which vectors are displayed

    show_gravity_manual_mass = flags[0]
    show_gravity = flags[2]   # Set to True to display gravity vectors, False to hide
    show_aero = flags[2]     # Set to True to display aero force vectors, False to hide
    show_normals = flags[3]   # Set to True to display normal vectors, False to hide


    normmult = 3 # Multiply norm vector to make it visible in animation

    resizeall = 0.025 # Resize all vectors of the animation to make them more or less fully visible.
    resize_torque_spiral = 0.01 # Resize torque spiral to make it more or less fully visible.

    zoom_resize = 0.5 # In order to make the frame of the animation more easily adjustable

    
    # Initial Position
    start_x, start_y = 0, 90

    # Initial Rotation
    initial_angle = initial_angle_in
    in_degrees = True

    inp = np.array(inp_in, dtype=float)
    currpoints = np.array(help.initialize_points(start_x, start_y, inp))
    com = np.array(help.compute_com_from_points(currpoints))
    
    # Shift of the manual point mass
    com_shift = np.array(com_shift_in, dtype=float)

    
    # Add manual mass and update COM accordingly
    if is_manual_mass:
        manual_pos = np.array(help.compute_com_from_points(currpoints)) + com_shift_in
        com = np.array(help.compute_com_from_points_with_mass(currpoints, manual_pos, manual_mass, mass_scale))
        com_shift = manual_pos - com

    com_init = com.copy()
    
    # Inital rotation applied to the points
    currpoints = help.rotate_points_around_com_degrees(currpoints, initial_angle, com, in_degrees)
 
    
    if is_manual_mass:
        # Inital rotation applied to the manual_pos
        rotated_manual_pos = np.array(help.rotate_points_around_com_degrees(np.array([manual_pos]), initial_angle, com, in_degrees))

        # Initial rotation applied to the manual_pos shift
        manual_pos = rotated_manual_pos[0]
        com_shift = manual_pos - com

    # Segments will have changed as well because of the rotation
    currsegs = help.compute_segments_from_points(currpoints)

    # Discretization
    currsegs = help.discretization(currsegs, num_discretization)

    # Mass of the line (sum of segment lengths)
    mass = np.sum([help.scaled_segment_length(seg, mass_scale) for seg in inp])
    if is_manual_mass:
        mass = mass + manual_mass


    # Initial velocity (2D) and angular velocity (scalar)
    currv = np.array(initial_velocity, dtype=float)
    curr_ang_v = initial_ang_velocity

    # Iteration counter and data recorders
    iteration = 0
    time_data = []
    vel_data = []
    angv_data = []


    # Initialize lists to store the last N velocities for scoring
    translational_velocities = []
    angular_velocities = []
    height_loss = []
    x_displacement = []
    aero_force_total = []

    translational_velocities.append(np.linalg.norm(currv))
    angular_velocities.append(curr_ang_v)
    height_loss.append(com_init[1]-com[1])
    x_displacement.append(com_init[0]-com[0])
    aero_force_total.append(np.zeros(2))


    # Number of timestamps to average for scores
    N = 50

    # ------------------------------------------
    # Create a 1×2 subplot with equal width spaces
    # ------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 6),
        gridspec_kw={'width_ratios': [1, 1]}
    )
    plt.tight_layout()

    # --------------------------------
    # Left subplot for line animation
    # --------------------------------

    # Compute bounding box from starting points
    currpoints_np = np.array(currpoints)
    x_vals = currpoints_np[:, 0]
    y_vals = currpoints_np[:, 1]

    xmin, xmax = np.min(x_vals), np.max(x_vals)
    ymin, ymax = np.min(y_vals), np.max(y_vals)

    # FIXED margins based only on initial size
    initial_width = xmax - xmin
    initial_height = ymax - ymin

    # Add generous margin
    xmargin = (xmax - xmin) * 6 * zoom_resize + 1
    ymargin_top = (ymax - ymin) * 2 * zoom_resize + 1
    ymargin_bottom = (ymax - ymin) * 10 * zoom_resize + 1

    xmargin_fixed = xmargin
    ymargin_top_fixed = ymargin_top
    ymargin_bottom_fixed = ymargin_bottom

    ax_left.set_aspect('equal', adjustable='box')
    ax_left.set_xlim(xmin - xmargin, xmax + xmargin)
    ax_left.set_ylim(ymin - ymargin_bottom, ymax + ymargin_top)


    ax_left.set_facecolor('white')
    ax_left.grid(True, which='both', linestyle='--', linewidth=0.5, color='lightgray')

    # Main line plot showing the segments
    line, = ax_left.plot([], [], 'b-', linewidth=2)
    # A star marker to show the center of mass each frame
    scom, = ax_left.plot([], [], marker='*', color='red', markersize=3, linestyle='')

    if is_manual_mass:
        smanual_pos,= ax_left.plot([], [], marker='*', color='black', markersize=3, linestyle='')

    # Initialize quiver objects for force vectors (gravity and aero forces)
    gravity_forces = [ax_left.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005) for _ in range(len(currsegs))]
    aero_forces = [ax_left.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='green', width=0.005) for _ in range(len(currsegs))]

    # Initialize a quiver for normals 
    normals = [ax_left.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='purple', width=0.005) for _ in range(len(currsegs))]

    if is_manual_mass:
        # Initialize a quiver for gravity force at manual_pos
        gravity_at_manual_pos = ax_left.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='blue', width=0.005)

    # Initialize torque spiral and quiver at the COM
    torque_spiral, = ax_left.plot([], [], 'orange', label="Torque Spiral")
    torque_quiver = ax_left.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='orange', width=0.005, label="Torque Spiral")


    # ---------------------------------------
    # Right subplot: velocity tracking plots
    # ---------------------------------------
    ax_right.set_xlabel('Time (s)')
    ax_right.set_ylabel('Spatial Velocity', color='orange')
    ax_right.tick_params(axis='y', labelcolor='orange')

    ax2 = ax_right.twinx()
    ax2.set_ylabel('Angular Velocity', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Disable scientific notation on both y-axes
    ax_right.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax_right.ticklabel_format(style='plain', axis='y')
    ax_right.get_yaxis().get_major_formatter().set_useOffset(False)

    ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=False))
    ax2.ticklabel_format(style='plain', axis='y')
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)

    # Lines for velocity vs. time
    v_line, = ax_right.plot([], [], color='orange', label='Spatial Velocity')
    angv_line, = ax2.plot([], [], color='blue', label='Angular Velocity')

    ax_right.legend([v_line, angv_line],
                    [v_line.get_label(), angv_line.get_label()],
                    loc='upper left')

    ax_right.set_facecolor('white')
    ax_right.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.set_facecolor('white')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax_right.set_xlim(0, 10)
    ax2.set_xlim(0, 10)

    # ---------------------------------------
    # ANIMATION INIT
    # ---------------------------------------


    def init():
        """
        Initialize the line objects before the animation starts.
        This function clears or resets any existing data.
        """
        # Reset the line plot data
        line.set_data([], [])
        
        # Reset the center of mass marker (COM)
        scom.set_data([], [])
        if is_manual_mass:
            smanual_pos.set_data([], [])
        
        # Reset the velocity and angular velocity data
        v_line.set_data([], [])
        angv_line.set_data([], [])
        
        if is_manual_mass:
            # Reset gravity vector at COM (only when is_manual_mass is True)
            gravity_at_manual_pos.set_offsets([0, 0])  # Reset position of the gravity vector at COM
            gravity_at_manual_pos.set_UVC(0, 0)  # Hide gravity vector at COM initially
        
        # Reset gravity, aero, and normal force vectors for all segments
        for gq, fq, nq in zip(gravity_forces, aero_forces, normals):
            gq.set_offsets([0, 0])  # Reset position of gravity force vectors
            gq.set_UVC(0, 0)  # Hide gravity vectors initially
            
            fq.set_offsets([0, 0])  # Reset position of aero force vectors
            fq.set_UVC(0, 0)  # Hide aero vectors initially
            
            nq.set_offsets([0, 0])  # Reset position of normal vectors
            nq.set_UVC(0, 0)  # Hide normal vectors initially
        
        # Reset torque spiral
        torque_spiral.set_data([], [])
		# Reset torque quiver (removes any previous arrows)
        torque_quiver.set_offsets([0, 0])
        torque_quiver.set_UVC(0, 0)

        if is_manual_mass:
            return line, scom, smanual_pos, v_line, angv_line, gravity_at_manual_pos, torque_spiral, torque_quiver, *gravity_forces, *aero_forces, *normals
        else:
            return line, scom, v_line, angv_line, torque_spiral, torque_quiver, *gravity_forces, *aero_forces, *normals


    # ---------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------

    def update(frame):
        """
        Update function called at each frame of the animation.
        Computes all forces, updates positions and velocities,
        and modifies the plot data accordingly.
        """
        global currpoints, currsegs, currv, curr_ang_v, iteration, translational_velocities, angular_velocities, com_shift, manual_pos


        # End the animation once we reach max_iteration
        if iteration >= max_iteration:
            plt.close(fig)
            if is_manual_mass:
                return line, scom, smanual_pos, v_line, angv_line, gravity_at_manual_pos, torque_spiral, torque_quiver,*gravity_forces, *aero_forces, *normals
            else:
                return line, scom, v_line, angv_line, torque_spiral, torque_quiver,*gravity_forces, *aero_forces, *normals


        # Compute the center of mass
        com = np.array(help.compute_com_from_points(currpoints))
        if is_manual_mass:
            com = np.array(help.compute_com_from_points_with_mass(currpoints, manual_pos, manual_mass, mass_scale))


        # Reset forces and torque
        force_total = np.zeros(2)
        force_current = np.zeros(2)
        torque_total = 0.0
        gravity_total = 0
        aero_total = np.zeros(2)

        # Calculate forces and torques
        for i, (segment, pt) in enumerate(zip(currsegs, currpoints)):
            
            segment = np.array(segment)
            pt = np.array(pt)

            # compute forces
            force_gravity = np.array(help.compute_fg(segment, gravity_acceleration, mass_scale))
            
            # If the airflow is not blocked by any other segment, compute the aerodynamic force and forward the deflected airflow to the other segments, if they themselves are not blocked (this functionality is inside the compute_deflected_airflow_force function)
            if not help.is_segment_blocked(segment, pt, currv, currpoints, i):
                force_aero = np.array(help.compute_force(segment, pt, currv, curr_ang_v, com))

                if deflected_airflow:
                    deflected_force_aero = help.compute_deflected_airflow_force(
                                                    alpha=alpha,
                                                    v_object=currv,
                                                    ang_vel=curr_ang_v,
                                                    segment=segment,
                                                    pt=pt,
                                                    all_segments=currsegs,
                                                    all_points=currpoints,
                                                    com=com,
                                                    segment_index=i,
                                                    neighbour_level = 8)
                    force_aero += deflected_force_aero
                aero_total += force_aero
                # Update aero force vector for animation
                if show_aero:
                    aero_forces[i].set_UVC(resizeall*force_aero[0], resizeall*force_aero[1])  # Direction and magnitude of the arrow
            else:
                force_aero = np.array([0, 0])
                # Update aero force vector for animation
                if show_aero:
                    aero_forces[i].set_UVC(resizeall*force_aero[0], resizeall*force_aero[1])  # Direction and magnitude of the arrow
            

            # Normal vector (perpendicular to the segment) for animation
            segment_length = np.linalg.norm(segment)
            if segment_length == 0:
                continue
            normal = (normmult*-segment[1] / segment_length, normmult*segment[0] / segment_length)  # Perpendicular to segment

            # Update normal vector for animation
            if show_normals:
                normals[i].set_UVC(resizeall*normal[0], resizeall*normal[1])  # Direction and magnitude of the normal

            # Update gravity force vector for segments for animation
            if show_gravity:
                gravity_forces[i].set_UVC(resizeall*force_gravity[0], resizeall*force_gravity[1])  # Direction and magnitude of the arrow

            
            # Add forces and use the current total to compute the torque
            force_current = force_gravity + force_aero
            torque_current = help.compute_torque(segment, pt, force_current, com)
            
            gravity_total += force_gravity
            
            # Track total force and torque
            force_total += force_current
            torque_total += torque_current

        # Record aero force total

        aero_force_total.append(aero_total.copy())

        # Add gravity force and torque from the manual mass
        if is_manual_mass:
            force_manual_pos = np.array(gravity_acceleration) * manual_mass
            force_total += force_manual_pos
            torque_total += help.compute_torque_manual_pos(manual_pos, force_manual_pos, com)

        
        # Update translational velocity
        currv += (force_total / mass) * dtime

        # Moment of inertia and angular velocity update
        mom_inertia_points = help.compute_moment_of_inertia_from_points(currpoints, com)

        if is_manual_mass:
            mom_inertia_manual_pos = help.compute_moment_of_inertia_from_manual_pos(manual_pos, manual_mass, com)
        mom_inertia = mom_inertia_points
        if is_manual_mass:
            mom_inertia = mom_inertia_points + mom_inertia_manual_pos
        ang_acc = torque_total / mom_inertia
        curr_ang_v += ang_acc * dtime
        angle_rot = curr_ang_v * dtime

        # Calculate and copy translational velocity (magnitude) for scoring and plotting
        translational_velocity = np.linalg.norm(currv)
        translational_velocities.append(translational_velocity)

        # Copy absolut of angular velocity for scoring and plotting
        angular_velocity = curr_ang_v
        angular_velocities.append(angular_velocity)

        
        # Rotate points around center of mass
        rotated_pts = help.rotate_points_around_com(currpoints, angle_rot, com)
        currpoints = np.array(rotated_pts)

        # Rotate the manual mass around COM
        if is_manual_mass:
            rotated_manual_pos = np.array(help.rotate_points_around_com(np.array([manual_pos]), angle_rot, com))
            manual_pos = rotated_manual_pos[0]
            com_shift = manual_pos - com
        

        # Recompute segments
        currsegs = help.compute_segments_from_points(currpoints)

        # Translate points based on current velocity
        currpoints += currv * dtime
        if is_manual_mass:
            manual_pos += currv * dtime

        
        # Update COM to have the right COM for the animation

        com = np.array(help.compute_com_from_points(currpoints))
        if is_manual_mass:
            com = help.compute_com_from_points_with_mass(currpoints, manual_pos, manual_mass, mass_scale)

        # Copy height of shape for scoring and plotting
        height_loss.append(com_init[1]-com[1])
        x_displacement.append(com_init[0]-com[0])


        # Update the midpoints where the arrows should be added for the animation
        for i, (segment, pt) in enumerate(zip(currsegs, currpoints)):
            
            segment = np.array(segment)
            pt = np.array(pt)

            # Midpoint of the segment (where we will place the arrows)
            midpoint = (pt[0] + segment[0] / 2, pt[1] + segment[1] / 2)

            # Update normal vector
            if show_normals:
                normals[i].set_offsets(midpoint)  # Position of the normal vector
            # Update gravity force vector
            if show_gravity:
                gravity_forces[i].set_offsets(midpoint)  # Position of the arrow

            # Update aero force vector
            if show_aero:
                aero_forces[i].set_offsets(midpoint)  # Position of the arrow


        # Update line data on left subplot
        xvals = np.append(currpoints[:, 0], currpoints[0, 0])
        yvals = np.append(currpoints[:, 1], currpoints[0, 1])
        line.set_data(xvals, yvals)

        # Update star marker for COM
        scom.set_data([com[0]], [com[1]])
        if is_manual_mass:
            smanual_pos.set_data([manual_pos[0]], [manual_pos[1]])

        # Apply the gravity force at the COM
        if is_manual_mass and show_gravity_manual_mass:
            # Apply gravity at the center of mass if the manual mass flag is set
            gravity_vector = gravity_acceleration * manual_mass  # Gravity force vector
            gravity_at_manual_pos.set_offsets(manual_pos)  # Update the position of the gravity vector
            gravity_at_manual_pos.set_UVC(resizeall*gravity_vector[0], resizeall*gravity_vector[1])  # Update the direction and magnitude of the vector
        elif is_manual_mass and (not(show_gravity_manual_mass)):
            # Hide gravity force at COM when is_manual_mass is False
            gravity_at_manual_pos.set_UVC(0, 0)  # Hide the vector at the COM

        # Calculate torque spiral
        torque_magnitude = np.linalg.norm(torque_total)  # Magnitude of the torque
        torque_direction = np.sign(torque_total)  # Sign determines direction of the torque (clockwise or counterclockwise)
        
        # Compute the torque spiral updates
        torque_spiral_scale = resizeall*resize_torque_spiral
        t = np.linspace(0, 10, num=100)  # Parametric variable for spiral
        r = torque_magnitude * t * torque_spiral_scale  # Smaller spiral size
        if torque_direction > 0:
            x_spiral = com[0] + r * np.cos(t)  # X-coordinates of the spiral counterclockwise
            y_spiral = com[1] + r * np.sin(t)  # Y-coordinates of the spiral counterclockwise
        else:
            x_spiral = com[0] + r * np.cos(-t)  # X-coordinates of the spiral clockwise
            y_spiral = com[1] + r * np.sin(-t)  # Y-coordinates of the spiral clockwise

        # Update the torque spiral
        torque_spiral.set_data(x_spiral, y_spiral)

		# Update the torque quiver
        torque_quiver_scale = 3
        torque_quiver.set_offsets([x_spiral[-1], y_spiral[-1]])
        torque_quiver.set_UVC(resizeall*torque_quiver_scale*np.cos(np.arctan2(y_spiral[-1] - y_spiral[-2], x_spiral[-1] - x_spiral[-2])), resizeall*torque_quiver_scale*np.sin(np.arctan2(y_spiral[-1] - y_spiral[-2], x_spiral[-1] - x_spiral[-2])))
    

        # Record data for velocity plots
        current_time = iteration * dtime
        
        if current_time % 1 == 0:  # Every second
            # Get the last N translational and angular velocities
            last_translational_velocities = translational_velocities[-N:]
            last_angular_velocities = angular_velocities[-N:]
            
            # Translational score
            T_score = sum(last_translational_velocities)/N
            
            # Angular score
            A_score = sum(np.abs(v) for v in last_angular_velocities)/N

            # Track height score
            height_score = com[1]
            
            # Print the scores
            print(f"Time: {current_time:.2f}s | ITERATION: {iteration} | Translational Score: {T_score} | Angular Score: {A_score} | Height Score: {height_score:.3f}")

        time_data.append(current_time)
        vel_data.append(np.linalg.norm(currv))
        angv_data.append(curr_ang_v)

        v_line.set_data(time_data, vel_data)
        angv_line.set_data(time_data, angv_data)

        # Auto-rescale the right subplot
        ax_right.relim()
        ax_right.autoscale_view()
        ax2.relim()
        ax2.autoscale_view()

        # Extend x-limits if time exceeds 10s
        if current_time > 10:
            ax_right.set_xlim(0, current_time)
            ax2.set_xlim(0, current_time)


        # -------------------------
        # Auto-adjust camera bounds
        # -------------------------
        # Get bounds of the current object
        x_vals = currpoints[:, 0]
        y_vals = currpoints[:, 1]

        xmin, xmax = np.min(x_vals), np.max(x_vals)
        ymin, ymax = np.min(y_vals), np.max(y_vals)

        xmargin = xmargin_fixed
        ymargin_top = ymargin_top_fixed
        ymargin_bottom = ymargin_bottom_fixed

        # Current axis limits
        cur_xlim = ax_left.get_xlim()
        cur_ylim = ax_left.get_ylim()

        # Check if we need to shift the view (if object is outside current view)
        if xmin < cur_xlim[0] or xmax > cur_xlim[1] or ymin < cur_ylim[0] or ymax > cur_ylim[1]:
            ax_left.set_xlim((xmax - initial_width) - xmargin, xmax + xmargin)
            ax_left.set_ylim((ymax - initial_height) - ymargin_bottom, ymax + ymargin_top)

        iteration += 1

        if is_manual_mass:
            return line, scom, smanual_pos, v_line, angv_line, gravity_at_manual_pos, torque_spiral, torque_quiver, *gravity_forces, *aero_forces, *normals
        else:
            return line, scom, v_line, angv_line, torque_spiral, torque_quiver, *gravity_forces, *aero_forces, *normals


    # Create and start the animation
    if plot_show:
        ani = animation.FuncAnimation(
            fig, update, frames=range(max_iteration),
            init_func=init, interval=1, repeat=False, blit=False
        )
        plt.show()
    else:
        # Run the simulation loop manually without animation
        for frame in range(max_iteration):
            update(frame)
    

    # Save the animation as a GIF using Pillow
    # ani.save('C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\BAfalling_object_simulation.gif', writer='pillow', fps=20)

    return {
    'time': time_data,
    'translational': translational_velocities[:-1],
    'rotational': angular_velocities[:-1],
    'height': height_loss[:-1],
    'x_displacement': x_displacement[:-1],
    'aero_force_total': aero_force_total[:-1]
}

