# Shape Analysis/Optimization of Falling Objects in 2D – Bachelor's Thesis Repository

## Abstract

This thesis presents a computational model for simulating the dynamics of 2D bodies under the influence of gravity and aerodynamic forces such as lift and drag. The simulation is inspired by paragliding, where subtle geometric changes in the wing — such as line trimming — can noticeably alter flight behavior. A lightweight physics-based model is implemented, which only computes aerodynamic forces for the segments of a shape locally instead of a global computational fluiddynamics simulation, to analyze how different 2D shape configurations behave during free fall. The model incorporates key concepts from rigid body dynamics and fluid dynamics, including center of mass, torque, moment of inertia, and force coefficients. Using this framework, various shape families are systematically varied and evaluated to assess their aerodynamic characteristics and overall performance during free fall. The thesis explores whether the resulting insights from a model with this local approach are plausible and thus could help inform both the design of flying structures and the understanding of how shape deformation affects stability and trajectory.

---

## Repository Overview

This repository contains the full simulation framework and analysis scripts from the Bachelor's thesis project. It enables the user to generate 2D shapes, simulate their aerodynamic descent, and evaluate their dynamic behavior under various physical configurations.

### Files Included

| File | Description |
|------|-------------|
| `BA.py` | Main simulation engine. Computes forces, updates object state, and produces Matplotlib animations. |
| `BAhelp.py` | Physics, math, and plotting utilities and helper functions for center of mass, forces, torques, and visualizations. |
| `BAwrapper.py` | Runs batch simulations and comparative experiments with different shape families. Includes all test cases/analyses from the thesis. |
| `BAwrapperhelp.py` | Shape generator library for ellipses, NACA profiles, paraglider-like shapes, umbrellas, and other custom forms. |

---

## Simulation parameters overview

| Parameter               | Description |
|------------------------|-------------|
| `inp_in`               | Input shape segments (defines the geometry of the body). |
| `com_shift_in`         | Manual shift vector in order to implement manual point mass ("jumper"), which will induce a new center of mass. |
| `initial_angle_in`     | Initial rotation angle of the shape in degrees. |
| `initial_velocity`     | Initial translational velocity as a 2D vector `[vx, vy]`. |
| `initial_ang_velocity` | Initial angular velocity (scalar). |
| `is_manual_mass`       | If `True`, adds a custom manual point mass to the body. |
| `manual_mass`          | Value of the manually added point mass. |
| `mass_scale`           | Scaling factor for mass contribution from each segment. A real paraglider is quite light, around 4kg |
| `flags`                | A list controlling visualization options: <br> `[manual gravity, _, segment gravity, normal vectors]` <br> Example: `[True, True, True, False]`. |
| `deflected_airflow`    | If `True`, enables airflow deflection modeling between segments. |
| `num_discretization`   | Number of subdivisions per segment for better accuracy. |
| `max_iteration`        | Maximum number of simulation steps (frames). |
| `dtime`                | Time step per frame (simulation granularity). |
| `alpha`                | Controls strength of airflow deflection between segments. |
| `gravity_acceleration` | Gravity vector (default: `[0.0, -9.81]`). |
| `plot_show`            | If `True`, shows an animation of the simulation. If `False`, runs headlessly and returns data. |



---

## Getting Started

### Requirements

Install Python packages via:

```bash
pip install numpy matplotlib shapely
```

### Run a custom Simulation

Edit the bottom of `BAwrapper.py` to pick a shape and set the parameters, then:

```bash
python BAwrapper.py
```

### Run Analyses

Uncomment the relevant analysis block(s) in `BAwrapper.py` to reproduce figures and tables from the thesis.

---

## Acknowledgments

This thesis was conducted under the supervision of:

- **Prof. Dr. Olga Sorkine-Hornung**
- **Marcel Padilla**
- **Aviv Segall**

Their guidance and insights were invaluable throughout the research and development of this simulation framework. I am sincerely grateful that they granted me the opportunity to work on this project.

---

## Contact

For questions, improvements, or collaboration inquiries, feel free to open an issue or contact me directly via GitHub.
