""" A set of robotics control functions """

import random
import numpy as np


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    laser_dist = lidar.get_sensor_values()
    laser_angles = lidar.get_ray_angles()

    window = np.pi
    window_dist = laser_dist[(laser_angles >= -window/2) & (laser_angles <= window/2)]
    min_dist = np.min(window_dist)

    if min_dist <= 10.0:
        speed = 0.0
        rotation_speed = 1.0
    else:
        speed = 1.0
        rotation_speed = 0.0

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command

def segment_lidar_clusters(ranges, angles, max_range, safe_dist, gap_threshold=20.0, min_cluster_size=2):

    # Filter 
    mask = (ranges < max_range - 1.0) & (ranges < safe_dist)
    if not np.any(mask):
        return []
 
    ranges = ranges[mask]
    angles = angles[mask]
 
    # Coordinate change
    x = np.cos(angles) * ranges
    y = np.sin(angles) * ranges
    pts = np.stack([x, y], axis=1)   
 
    # Segmentation
    sort_idx = np.argsort(angles)
    pts = pts[sort_idx]
    r_sorted = ranges[sort_idx] # Sorted by angle
 
    clusters = []
    current_pts = [pts[0]]
    current_r   = [r_sorted[0]]
 
    for i in range(1, len(pts)):
        # Distance euclidienne
        d = np.linalg.norm(pts[i] - pts[i - 1])
 
        jump_in_range = abs(r_sorted[i] - r_sorted[i - 1]) > gap_threshold * 0.5
 
        if d > gap_threshold or jump_in_range: # Agrupper entre les points proches
            if len(current_pts) >= min_cluster_size:
                arr = np.array(current_pts)
                clusters.append({
                    'points_xy': arr,
                    'centroid':  arr.mean(axis=0),
                    'min_dist':  np.min(current_r),
                })
            # New cluster
            current_pts = [pts[i]]
            current_r   = [r_sorted[i]]
        else:
            current_pts.append(pts[i])
            current_r.append(r_sorted[i])
 
    if len(current_pts) >= min_cluster_size:
        arr = np.array(current_pts)
        clusters.append({
            'points_xy': arr,
            'centroid':  arr.mean(axis=0),
            'min_dist':  np.min(current_r),
        })
 
    return clusters


def potential_field_control(lidar, current_pose, goal_pose, stop_dist: float = 20.0):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    stop_dist : distance threshold at which the robot stops for the current goal
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2

    # Parameters
    K_goal = 0.5
    K_obst = 8000
    safe_dist = 50.0
    d_trans = 100.0

    GAP_THRESHOLD    = 20.0
    MIN_CLUSTER_SIZE = 2

    curr_p = current_pose[:2]
    goal_p = goal_pose[:2]
    theta = current_pose[2]

    diff = goal_p - curr_p
    goal_dist = np.linalg.norm(diff)
    
    # Tolerance to goal
    if goal_dist < stop_dist:
        return {"forward": 0.0, "rotation": 0.0}
  
    # Gradient Attractif 
    if goal_dist > d_trans:
        # Linéaire : Vitesse constante
        grad_attr = K_goal * diff / goal_dist
    else:
        # Quadratique : Vitesse proportionnelle à la distance
        grad_attr = K_goal * diff / d_trans

    # Changement de repère 
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, s], [-s, c]])
    grad_attr = rot_matrix @ grad_attr

    # Gradient Répulsif 
    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    clusters = segment_lidar_clusters(ranges, angles, max_range=lidar.max_range, safe_dist=safe_dist, gap_threshold=GAP_THRESHOLD,min_cluster_size=MIN_CLUSTER_SIZE,)

    grad_rep = np.zeros(2)

    for cluster in clusters:
        d_min = cluster['min_dist']
        centroid = cluster['centroid']
        dist_centroid = np.linalg.norm(centroid)

        if dist_centroid < 1e-3:
            continue  # avoid division by zero

        direction = centroid / dist_centroid

        if d_min < safe_dist and d_min > 0.1:
            mag = K_obst * (1.0/d_min - 1.0/safe_dist) * (1.0/d_min**2)
            grad_rep -= mag * direction

    grad_total = grad_attr + grad_rep

    v = grad_total[0] 
    w = np.arctan2(grad_total[1], grad_total[0]) * 0.5

    # Clamp values to valid ranges [-1, 1]
    v = np.clip(v, -1, 1)
    w = np.clip(w, -0.1, 0.1)
    
    command = {"forward": v,
               "rotation": w}
    
    return command


