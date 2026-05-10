# ensta-rob201 — ROB201 Mobile Robotics Project

This repository is a student implementation of the ROB201 course project at ENSTA Paris. The base code and simulator are available at: [GitHub repository *ensta-rob201*](https://github.com/emmanuel-battesti/ensta-rob201).

The project is built on top of the **Place-Bot** simulator: [**Place-Bot** GitHub repository](https://github.com/emmanuel-battesti/place-bot).

---

## Project structure

```
ensta-rob201/
├── main.py               # Entry point — configures and launches the simulator
├── my_robot_slam.py      # Main robot controller (SLAM + planning + exploration)
├── control.py            # Low-level control functions (reactive avoidance, potential field)
├── tiny_slam.py          # TinySlam: occupancy grid mapping and localisation
├── occupancy_grid.py     # Occupancy grid class (ray tracing, display, save/load)
├── planner.py            # Path planner (A*) and frontier-based exploration
└── worlds/
    └── my_world.py       # Simulation environment definition
```

---

## What was implemented

### Session 1 — Reactive obstacle avoidance (`control.py`)

A simple reactive controller (`reactive_obst_avoid`) that reads LiDAR data and commands the robot to move forward when the path is clear, or rotate when an obstacle is detected within a threshold distance.

### Session 2 — Potential field control (`control.py`)

A `potential_field_control` function that combines:

- an **attractive gradient** toward the goal (linear at long range, quadratic near the goal for smooth deceleration),
- a **repulsive gradient** from nearby obstacles, computed from LiDAR clusters segmented by `segment_lidar_clusters`.

### Session 3 — Occupancy grid mapping (`tiny_slam.py`)

The `update_map` function integrates each LiDAR scan into the occupancy grid using a Bayesian model:

- cells between the robot and each detection are marked as **free** (with a stronger weight near the robot),
- cells at the detection point are marked as **occupied** using a Gaussian kernel (`OCC_PEAK`, `OCC_SIGMA`, `OCC_WINGS`),
- all values are clipped to `[-CLIP_MAX, CLIP_MAX]` to prevent saturation.

An adaptive subsampling step (`_adaptive_subsample`) avoids redundant updates from rays that hit the same cell.

### Session 4 — Localisation (`tiny_slam.py`)

The `_score` function evaluates a candidate robot pose by summing the occupancy values at the LiDAR end-points — higher scores indicate better alignment with the existing map.

Localisation is implemented using the **Cross-Entropy Method (CEM)**: at each call, a population of pose offsets is sampled from a Gaussian distribution, evaluated with `_score`, and the distribution is re-fitted on the elite subset.

The corrected pose is computed by `get_corrected_pose`, which transforms the raw odometry position into the map frame using the current odometry reference `odom_pose_ref`.

### Session 5 — Path planning (`planner.py`)

The `plan` function implements **A\*** on the occupancy grid:

- obstacles are dilated with an elliptical kernel before planning, so the robot body is implicitly accounted for,
- goals that fall inside the inflated wall zone are rejected before search,
- the resulting path is converted back to world coordinates and followed waypoint by waypoint in `my_robot_slam.py`.

### Session 6 — Frontier-based exploration (`planner.py`)

Full **Frontier-Based Exploration** [Yamauchi, 1997]:

- `get_frontiers`: detects frontier cells — free cells adjacent to unknown cells — using vectorised NumPy array shifts. Cells within `wall_clearance_cells` of an obstacle are excluded via morphological dilation, preventing the robot from targeting frontiers that are too close to walls.
- `cluster_frontiers`: groups frontier cells into spatially coherent clusters. Clusters smaller than `min_cluster_size` are discarded as noise.
- `select_best_frontier`: ranks clusters by a weighted utility score combining normalised cluster size and normalised distance to the robot, and returns the centroid of the best cluster as the next goal.

#### Return to start

Once no more frontiers are detected (exploration complete), the robot automatically plans an A\* path back to its starting position `[0, 0, 0]` and follows it to completion.

---

## Contact

Questions about the base simulator: emmanuel.battesti@ensta-paris.fr