import numpy as np
from BA import run_simulation
import BAwrapperhelp as wrapperhelp
import BAhelp as help
from multiprocessing import Process


# # -----------------------------
# # Experiments (Example usage for own shapes at the end)
# # -----------------------------

# # -----------------------------
# # Translational Analyses (4)
# # -----------------------------

# # -----------------------------
# # Analysis 1, Lower Ellipse that is "cut out" is bigger and shapes are not scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=5.5, d=4.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 101
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save= True, title= "Bigger cutout not scaled")




# # -----------------------------
# # Analysis 2, Lower Ellipse that is "cut out" is smaller and shapes are not scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=4.5, d=3.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 101
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save= True, title= "Smaller cutout not scaled")





# # -----------------------------
# # Analysis 3, Lower Ellipse that is "cut out" is bigger and shapes are scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses (scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=5.5, d=4.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 101
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save= True, title= "Bigger cutout scaled")



# # -----------------------------
# # Analysis 4, Lower Ellipse that is "cut out" is smaller and shapes are scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses (scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=4.5, d=3.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 101
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save = True, title= "Smaller cutout scaled")


















# # -----------------------------
# # Angular Analyses (4)
# # -----------------------------


# # -----------------------------
# # Analysis 5, Lower Ellipse that is "cut out" is bigger and shapes are not scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=5.5, d=4.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save= True, title= "Bigger cutout not scaled inital angle")




# # -----------------------------
# # Analysis 6, Lower Ellipse that is "cut out" is smaller and shapes are not scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=4.5, d=3.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save= True, title= "Smaller cutout not scaled inital angle")





# # -----------------------------
# # Analysis 7, Lower Ellipse that is "cut out" is bigger and shapes are scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses (scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=5.5, d=4.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save= True, title= "Bigger cutout scaled inital angle")



# # -----------------------------
# # Analysis 8, Lower Ellipse that is "cut out" is smaller and shapes are scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses (scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=4.5, d=3.5, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save = True, title= "Smaller cutout scaled inital angle")

































# # -----------------------------
# # Analysis 9, Same Size Ellipse that is "cut out" shapes are scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses, with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=5, d=4, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 31
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # Assume results = [Overlap 0.0, Overlap 0.1, ..., Overlap 0.9]
# # Get time vector (all the same)
# time = results[0]['time']

# # Select only every second parameter
# results_filtered = results[::2]  # Take only every second result: 0, 2, 4, etc.

# # Start LaTeX table
# print("\\begin{table}[H]")
# print("\\centering")
# print("\\scriptsize")  # Make table smaller horizontally
# print("\\setlength{\\tabcolsep}{3pt}")  # Smaller column spacing
# print("\\renewcommand{\\arraystretch}{1.2}")  # Normal vertical spacing
# print("\\begin{tabular}{c|" + "c" * len(results_filtered) + "}")
# print("\\toprule")
# print("Time (s) & " + " & ".join([f"{res['name']} T" for res in results_filtered]) + "\\\\")
# print("\\midrule")

# # Fill first 26 rows
# for i in range(26):
#     t = time[i]
#     translational_scores = [res['translational'][i] for res in results_filtered]

#     line = f"{t:.2f}"
#     for score in translational_scores:
#         line += f" & {score:.2f}"
#     line += " \\\\"
#     print(line)

# print("\\bottomrule")
# print("\\end{tabular}")
# print("\\caption{Translational velocities (T) over time for selected overlaps (every second parameter) for the identical cutout case.}")
# print("\\label{tab:same_cutout_scaled_filtered}")
# print("\\end{table}")







# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save = True, title= "Same cutout scaled")










# # -----------------------------
# # Analysis 10, Same Size Ellipse that is "cut out" shapes are scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses, with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_scaled_ellipse_diff(param=p, a=5, b=4, c=5, d=4, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save = True, title= "Same cutout scaled initial angle")












# # -----------------------------
# # Analysis 11 Half Circle
# # -----------------------------

# # -----------------------------
# # Generate paramtrized shapes between Half Circle and "Paraglider/Crescent Moon"
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff_flat_base(param=p, a=5, b=4, c=5, d= 0, num_points=50) for p in params]

# # -----------------------------
# # Run Simulations and Collect Scores
# # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):

#     sim_result = run_simulation(
#         inp_in=shape, initial_angle_in = 0, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 101)
    

#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })

# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], plot_show = False, save= True, title="Half circle")

















# # -----------------------------
# # Analysis 12 Deflected Airflow
# # -----------------------------

# # -----------------------------
# # Generate the two shapes
# # -----------------------------

# deflected_airflow_one = np.array([[2, 0], [0, -2], [-1, 1], [-1, -1], [0, 2]])
# # help.plot_shape_deflected_airflow(deflected_airflow_one, save=True, save_path= r"C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\deflected_airflow_one.png")

# deflected_airflow_two = np.array([[2, 0], [0, -2], [-1, -1], [-1, 1], [0, 2]])
# # help.plot_shape_deflected_airflow(deflected_airflow_two, save=True, save_path = r"C:\Daten_Lokal\Daten\Ausbildung\Studium\BAgifs\deflected_airflow_two.png")

# shapes = [deflected_airflow_one, deflected_airflow_two]
# names1 = ["Open_ND", "Spear_ND"]
# names2 = ["Open_D", "Spear_D"]

# # -----------------------------
# # Run Simulations and Collect Scores
# # -----------------------------
# results = []

# # Run the basic simulation

# for idx, shape in enumerate(shapes):

#     sim_result = run_simulation(
#         inp_in=shape, plot_show = False, deflected_airflow = False, max_iteration = 101)
    
#     results.append({
#         "name": f"{names1[idx]}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height'],
#         "aero_force_total": sim_result['aero_force_total']
#     })

# # Run the simulation with the deflected airflow forces

# for idx, shape in enumerate(shapes):

#     sim_result = run_simulation(
#         inp_in=shape, plot_show = False, deflected_airflow = True, alpha = 0.2, max_iteration = 101)
    
#     results.append({
#         "name": f"{names2[idx]}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height'],
#         "aero_force_total": sim_result['aero_force_total']
#     })


# # Assume results = [Open_ND, Spear_ND, Open_D, Spear_D]
# # Get time vector (all the same)
# time = results[0]['time']

# # Start LaTeX table
# print("\\begin{table}[H]")
# print("\\centering")
# print("\\scriptsize")  # Make the text slightly smaller
# print("\\setlength{\\tabcolsep}{3pt}")  # Reduce only horizontal column spacing
# print("\\begin{tabular}{c|cccc|cccc|c|cccc}")
# print("\\toprule")
# print("Time (s) & \\multicolumn{4}{c|}{Transl. Velocity (m/s)} & \\multicolumn{4}{c|}{Height Loss (m)} & Aeroforce (N) & \\multicolumn{4}{c}{Aero Force y (N)}\\\\")
# print(" & Spear ND & Open ND & Spear D & Open D & Spear ND & Open ND & Spear D & Open D & All x-comp. & Spear ND & Open ND & Spear D & Open D\\\\")
# print("\\midrule")

# # Fill each row
# for i in range(len(time)):
#     t = time[i]
#     translational_scores = [res['translational'][i] for res in results]
#     height_scores = [res['height'][i] for res in results]
#     aero_forces = [res['aero_force_total'][i] for res in results]  # Keep full vectors [x,y]

#     line = f"{t:.2f}"
#     for score in translational_scores:
#         line += f" & {score:.2f}"
#     for score in height_scores:
#         line += f" & {score:.2f}"
#     # Print x-component once (same for all, usually 0)
#     line += f" & {aero_forces[0][0]:.2f}"
#     # Print all y-components separately
#     for force in aero_forces:
#         line += f" & {force[1]:.2f}"
#     line += "\\\\"
#     print(line)

# print("\\bottomrule")
# print("\\end{tabular}")
# print("\\caption{Full translational velocities, heights, and aerodynamic force components over time for both shapes with and without deflected airflow.}")
# print("\\label{tab:deflected_airflow_full_with_aero_components}")
# print("\\end{table}")


# # -----------------------------
# # Plot All Scores
# # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], plot_show = True, save= True, title="Deflected Airflow longer")











# -----------------------------
# Analysis 13 Different Cutouts
# -----------------------------

# -----------------------------
# Generate the three shapes
# -----------------------------


help.plot_shape(wrapperhelp.generate_ellipse_above_diff(param=0.5, a=9, b=6, c=5, d=2, num_points=50, gap=1.5)
)
# help.plot_shape(wrapperhelp.generate_ellipse_above_diff(param=0.5, a=9, b=6, c=9, d=6, num_points=50, gap=0)
# )
# help.plot_shape(wrapperhelp.generate_ellipse_above_diff(param=0.5, a=9, b=6, c=13, d=10, num_points=50, gap=0)
# )




# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], save = True, title= "Same cutout not scaled")












# # -----------------------------
# # Analysis 14 Deflected airflow with more complex shapes and an initial angle
# # -----------------------------

# # -----------------------------
# # Generate the two shapes
# # -----------------------------

# params = [0.0, 0.7]
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=5, d=4, num_points=20, gap=0) for p in params]
# names1 = ["Ellipse_ND", "Cutout_ND"]
# names2 = ["Ellipse_D", "Cutout_D"]

# # -----------------------------
# # Run Simulations and Collect Scores
# # -----------------------------
# results = []

# # Run the basic simulation

# for idx, shape in enumerate(shapes):

#     sim_result = run_simulation(
#         inp_in=shape, initial_angle_in=30, plot_show = False, deflected_airflow = False, max_iteration = 101)
    
#     results.append({
#         "name": f"{names1[idx]}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height'],
#         "aero_force_total": sim_result['aero_force_total']
#     })

# # Run the simulation with the deflected airflow forces

# for idx, shape in enumerate(shapes):

#     sim_result = run_simulation(
#         inp_in=shape, initial_angle_in=30, plot_show = False, deflected_airflow = True, alpha = 0.2, max_iteration = 101)
    
#     results.append({
#         "name": f"{names2[idx]}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height'],
#         "aero_force_total": sim_result['aero_force_total']
#     })


# # Assume results = ["Ellipse_ND", "Cutout_ND", "Ellipse_D", "Cutout_D"]
# # Get time vector (all the same)
# time = results[0]['time']

# # Start LaTeX table
# print("\\begin{table}[H]")
# print("\\centering")
# print("\\scriptsize")  # Make the text slightly smaller
# print("\\setlength{\\tabcolsep}{3pt}")  # Reduce only horizontal column spacing
# print("\\begin{tabular}{c|cccc|cccc|c|cccc}")
# print("\\toprule")
# print("Time (s) & \\multicolumn{4}{c|}{Transl. Velocity (m/s)} & \\multicolumn{4}{c|}{Height Loss (m)} & Aeroforce (N) & \\multicolumn{4}{c}{Aero Force y (N)}\\\\")
# print(" & Spear ND & Open ND & Spear D & Open D & Spear ND & Open ND & Spear D & Open D & All x-comp. & Spear ND & Open ND & Spear D & Open D\\\\")
# print("\\midrule")

# # Fill each row
# for i in range(26):
#     t = time[26]
#     translational_scores = [res['translational'][i] for res in results]
#     height_scores = [res['height'][i] for res in results]
#     aero_forces = [res['aero_force_total'][i] for res in results]  # Keep full vectors [x,y]

#     line = f"{t:.2f}"
#     for score in translational_scores:
#         line += f" & {score:.2f}"
#     for score in height_scores:
#         line += f" & {score:.2f}"
#     # Print x-component once (same for all, usually 0)
#     line += f" & {aero_forces[0][0]:.2f}"
#     # Print all y-components separately
#     for force in aero_forces:
#         line += f" & {force[1]:.2f}"
#     line += "\\\\"
#     print(line)

# print("\\bottomrule")
# print("\\end{tabular}")
# print("\\caption{Full translational velocities, heights, and aerodynamic force components over time for both shapes with and without deflected airflow.}")
# print("\\label{tab:deflected_airflow_full_with_aero_components}")
# print("\\end{table}")


# # -----------------------------
# # Plot All Scores
# # -----------------------------

# help.plot_simulation_scores(results, show=[True, False, False], plot_show = True, save= True, title="Deflected Airflow with initial angle")












# # -----------------------------
# # Analysis 15, Same Size Ellipse that is "cut out" shapes are not scaled
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=5, d=4, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save = True, title= "Same cutout not scaled")









# # -----------------------------
# # Analysis 16, Same Size Ellipse that is "cut out" shapes are not scaled, inital angle of 30 degrees
# # -----------------------------

# # -----------------------------
# # Generate ellipses (not scaled), with varying overlap (param 0.0 to 0.9)
# # -----------------------------

# params = np.linspace(0.0, 0.9, 17)
# shapes = [wrapperhelp.generate_ellipse_above_diff(param=p, a=5, b=4, c=5, d=4, num_points=50, gap=0) for p in params]

# # # -----------------------------
# # # Run Simulations and Collect Scores
# # # -----------------------------
# results = []

# for idx, (shape, param) in enumerate(zip(shapes, params)):
#     sim_result = run_simulation(
#         inp_in=shape, initial_velocity = np.array([0.0, -15.0]), initial_angle_in = 30, plot_show = False, max_iteration = 401
#     )
    
#     results.append({
#         "name": f"Overlap {param:.1f}",
#         "time": sim_result['time'],
#         "translational": sim_result['translational'],
#         "rotational": sim_result['rotational'],
#         "height": sim_result['height']
#     })



# # # -----------------------------
# # # Plot All Scores
# # # -----------------------------

# help.plot_simulation_scores(results, show=[True, True, False], save = True, title= "Same cutout not scaled initial angle")





















# -----------------------------
# Run your own simulation
# -----------------------------


# -----------------------------
# Simulation parameters & setup
# -----------------------------

max_iteration = 101

# Vector display flags: [gravity_manual_mass, gravity, aero, normals]
flags = [True, True, True, False]

is_manual_mass = True
manual_mass = 100

num_discretization = 2

# --------------------------------------------------
# Select shape generator function (scale if needed)
# --------------------------------------------------

# Example shapes to choose from:
# inp = wrapperhelp.generate_naca_0012_truncated(scale=1)
# inp = wrapperhelp.generate_symmetrical_umbrella(scale=1)
# inp = wrapperhelp.generate_asymmetrical_umbrella(scale=1)
# inp = wrapperhelp.generate_alternate_umbrella(scale=1)
inp = wrapperhelp.generate_paraglider(scale=2)
# inp = wrapperhelp.generate_ellipse(a=15, b=5, c=15, d=5, num_points=50)
# inp = help.compute_segments_from_points(ellipse_points)
# inp = wrapperhelp.generate_star(5, 2, 4)
# inp = wrapperhelp.generate_rectangle(4, 10)
# inp = wrapperhelp.generate_triangle(4, 10)

# -----------------------------
# Run Simulation
# -----------------------------

def run_one():
    run_simulation(inp, initial_angle_in=0, max_iteration = max_iteration, flags = flags, is_manual_mass = is_manual_mass, manual_mass = manual_mass, num_discretization = num_discretization
)

def run_two():
    run_simulation(inp, initial_angle_in=45, max_iteration = max_iteration, flags = flags, is_manual_mass = is_manual_mass, manual_mass = manual_mass, num_discretization = num_discretization
)

if __name__ == "__main__":

    p1 = Process(target=run_one)
    p2 = Process(target=run_two)
    p1.start()
    # p2.start()

