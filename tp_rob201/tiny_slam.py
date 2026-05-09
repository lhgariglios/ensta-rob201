""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid

FREE_NEAR   = -1.5 # free near to the robot
FREE_FAR    = -0.3 # free far from the robot
 
OCC_PEAK    =  3.0 # obstacle
OCC_SIGMA   =  1.2 # cells around the obstacle
OCC_WINGS   =  2   # neighbor cells updated
 
CLIP_MAX    = 20.0 # cell saturation
 
MIN_CELL_DIST = 1  # subsample

# Consider only cells at least this far from the robot for map update and scoring, to avoid biasing the map with the same cells at each scan

def _adaptive_subsample(ranges, angles, pose, grid, min_cell_dist=MIN_CELL_DIST):

    if len(ranges) == 0:
        return ranges, angles
 
    x_end = np.cos(angles + pose[2]) * ranges + pose[0]
    y_end = np.sin(angles + pose[2]) * ranges + pose[1]
 
    xm, ym = grid.conv_world_to_map(x_end, y_end)
 
    kept = np.zeros(len(ranges), dtype=bool)
    last_xm, last_ym = -9999, -9999
 
    for i in range(len(ranges)):
        if max(abs(int(xm[i]) - last_xm), abs(int(ym[i]) - last_ym)) >= min_cell_dist:
            kept[i] = True
            last_xm, last_ym = int(xm[i]), int(ym[i])
 
    return ranges[kept], angles[kept]


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
    
    def _sensor_end_points(self, lidar, pose):

        ranges = lidar.get_sensor_values()
        angles = lidar.get_ray_angles()

        mask = ranges < lidar.max_range - 20
        ranges, angles = ranges[mask], angles[mask]

        ranges, angles = _adaptive_subsample(ranges, angles, pose, self.grid)

        x_end = np.cos(angles + pose[2]) * ranges + pose[0]
        y_end = np.sin(angles + pose[2]) * ranges + pose[1]
        return x_end, y_end, ranges, angles

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        x, y, _, _ = self._sensor_end_points(lidar, pose)

        x_map, y_map = self.grid.conv_world_to_map(x, y)

        mask = (x_map > 0) & (x_map < self.grid.x_max_map) & (y_map > 0) & (y_map < self.grid.y_max_map)
        x_map = x_map[mask]
        y_map = y_map[mask]

        score = float(np.sum(self.grid.occupancy_map[x_map, y_map]))

        return score
    
    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        corrected_pose = odom_pose

        if (odom_pose_ref is None):
            odom_pose_ref = self.odom_pose_ref

        d0 = np.sqrt(odom_pose[0]**2 + odom_pose[1]**2)
        alpha0 = np.atan2(odom_pose[1], odom_pose[0])

        x = odom_pose_ref[0] + d0 * np.cos(odom_pose_ref[2] + alpha0)
        y = odom_pose_ref[1] + d0 * np.sin(odom_pose_ref[2] + alpha0)
        theta = odom_pose_ref[2] + odom_pose[2]

        corrected_pose = np.array([x, y, theta])

        return corrected_pose
    
    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        best_score = 0
        n = 200
        i = 0
        sigma = 1

        pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar,pose)
        best_pose_ref = self.odom_pose_ref.copy()

        while i < n:
            offset = np.random.normal(0,sigma,3)
            offset[2] = np.random.normal(0,sigma/10)
            pose_ref = best_pose_ref + offset
            pose = self.get_corrected_pose(raw_odom_pose,best_pose_ref)
            score = self._score(lidar,pose)
            if score > best_score:
                best_score = score
                best_pose_ref = pose_ref
            else:
                i += 1

        self.odom_pose_ref = best_pose_ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """ 

        x_list, y_list, ranges, angles = self._sensor_end_points(lidar, pose)

        rx, ry = pose[0], pose[1]

        for x, y, d in zip(x_list, y_list, ranges):
 
            if d > 1e-3:
                # Half of the path
                x_mid = rx + (x - rx) * 0.5
                y_mid = ry + (y - ry) * 0.5
 
                # Free near the robot
                self.grid.add_value_along_line(rx, ry, x_mid, y_mid,val=FREE_NEAR)

                # Free far from the robot
                self.grid.add_value_along_line(x_mid, y_mid, x, y, val=FREE_FAR)
 
            # Robot → Obstacle en coordonnées monde
            if d > 1e-3:
                ux = (x - rx) / d
                uy = (y - ry) / d
            else:
                continue
 
            step = self.grid.resolution # Cell correlation to real distance
 
            for k in range(-OCC_WINGS, OCC_WINGS + 1):
                # Gaussien
                weight = OCC_PEAK * np.exp(-0.5 * (k / OCC_SIGMA) ** 2)
 
                xk = x + k * step * ux
                yk = y + k * step * uy
 
                # Vérification if the point is in the map
                xmk, ymk = self.grid.conv_world_to_map(xk, yk)
                if (0 <= xmk < self.grid.x_max_map and 0 <= ymk < self.grid.y_max_map):
                    self.grid.occupancy_map[xmk, ymk] += weight
 
        # Saturation
        self.grid.occupancy_map = np.clip(
            self.grid.occupancy_map, -CLIP_MAX, CLIP_MAX
        )
 
        



