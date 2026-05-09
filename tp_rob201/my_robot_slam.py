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

        # path planning
        self.path = None
        self.path_index = 0
        self.current_goal = None

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

        # Localisation and mapping

        if self.counter > 10:
            self.tiny_slam.localise(self.lidar(), pose)

        pose = self.tiny_slam.get_corrected_pose(pose) 
        self.tiny_slam.update_map(self.lidar(), pose)

        # Frontier-based goal selection
        goal_reached = (
            self.current_goal is not None
            and np.linalg.norm(self.current_goal[:2] - pose[:2]) < 20.0
        )

        if self.current_goal is None or goal_reached:
            self.path = None
            self.current_goal = np.array([np.random.uniform(-100, 100), np.random.uniform(-100, 100), 0])
            #self.current_goal = self.planner.explore_frontiers(pose)
            self.path = self.planner.plan(pose, self.current_goal)
            if self.path is not None:
                self.path = self.path.T
                self.path_index = 0           

        #self.current_goal = np.array([-400, -100, 0])

        # Path planning and following

        if self.counter % 50 == 0:
            # Plan path to the new goal
            self.path = self.planner.plan(pose, self.current_goal)
            if self.path is not None:
                self.path = self.path.T
                self.path_index = 0
            else:
                self.path = None          

        if self.path is not None and self.path_index < len(self.path):
            target = self.path[self.path_index]

            if np.linalg.norm(target - pose[:2]) < 10.0:  # close to waypoint
                self.path_index += 1

            if self.path_index < len(self.path):
                target = self.path[self.path_index]
            else:
                target = self.current_goal[:2]  # end of path, go to goal

            target_pose = np.array([target[0], target[1], 0.0])
            command = potential_field_control(self.lidar(), pose, target_pose, stop_dist=10.0)

        else:
            # No path: go directly toward the goal
            target = self.current_goal[:2]
            target_pose = np.array([target[0], target[1], 0.0])
            command = potential_field_control(self.lidar(), pose, target_pose)

        
        # Display every 10 steps

        if self.counter % 10 == 0:
            traj = self.path.T if self.path is not None else None
            self.occupancy_grid.display_cv(pose, self.current_goal, traj)

        self.counter += 1

        return command
