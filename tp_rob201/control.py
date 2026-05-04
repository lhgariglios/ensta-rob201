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
    safe_dist = 20.0
    d_trans = 40.0

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

    # Gradient Répulsif 
    grad_rep = np.array([0.0, 0.0])
    ranges = lidar.get_sensor_values()
    angles = lidar.get_ray_angles()

    for d, angle in zip(ranges, angles):
        if 0.1 < d < safe_dist:
            mag = K_obst * (1.0/d - 1.0/safe_dist) * (1.0/d**2)
            grad_rep -= mag * np.array([np.cos(angle), np.sin(angle)])

    # Changement de repère 
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, s], [-s, c]])
    grad_attr = rot_matrix @ grad_attr

    grad_total = grad_attr + grad_rep

    v = grad_total[0] 
    w = np.arctan2(grad_total[1], grad_total[0]) * 0.5

    # Clamp values to valid ranges [-1, 1]
    v = np.clip(v, -0.3, 0.3)
    w = np.clip(w, -0.1, 0.1)
    
    command = {"forward": v,
               "rotation": w}
    
    return command


