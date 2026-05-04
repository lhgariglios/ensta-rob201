"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """

        return self.control_tp2()

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        #self.tiny_slam.compute()

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()

        if self.counter > 10:
            self.tiny_slam.localise(self.lidar(), pose)

        pose = self.tiny_slam.get_corrected_pose(pose) 
        self.tiny_slam.update_map(self.lidar(), pose)
 
        if not hasattr(self, 'current_goal'):

            self.current_goal = np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100), 0])

        elif np.linalg.norm(self.current_goal[:2] - pose[:2]) < 20.0:

            ranges = self.lidar().get_sensor_values()
            angles = self.lidar().get_ray_angles()

            mask = ranges < self.lidar().max_range+1.5
            angles = angles[mask]
            ranges = ranges[mask]

            idx = np.random.choice(len(ranges))
            distance = ranges[idx]

            safe_distance = 10.0
            
            if distance > safe_distance + 5.0:
                distance_goal = np.random.uniform(safe_distance, distance - 5.0)
            else:
                distance_goal = distance * 0.5

            ray_angle = angles[idx] + pose[2]

            x = pose[0] + distance_goal * np.cos(ray_angle)
            y = pose[1] + distance_goal * np.sin(ray_angle)

            self.current_goal = np.array([x, y, 0])

        command = potential_field_control(self.lidar(), pose, self.current_goal)

        self.counter += 1
        if self.counter % 10 == 0:
            self.occupancy_grid.display_cv(pose, self.current_goal)

        return command
