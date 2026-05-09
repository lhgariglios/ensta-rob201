"""
Planner class
Implementation of A*
"""


import copy
import heapq
import math
from collections import defaultdict
from typing import Optional,Tuple


import cv2
import numpy as np
from occupancy_grid import OccupancyGrid




class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid
        self.map_walls = None
   
    def get_neighbors(self, current_cell): # return the 8 neighbors of a cell
        x, y = current_cell
        x_max, y_max = self.grid.occupancy_map.shape

        neighbors = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx = x + dx
                ny = y + dy
                if 0 <= nx < x_max and 0 <= ny < y_max:
                    neighbors.append((nx, ny))
        return neighbors
   
    def heuristic(self, cell1, cell2): # euclidean distance between two cells
        x1, y1 = cell1
        x2, y2 = cell2
        d = np.sqrt( (x2-x1)**2 + (y2-y1)**2 )
        return d


    def reconstruct_path(self, came_from, goal):
        """ Extract path after cost computation """
        total_path = [goal]
        cell = goal
        while cell in came_from.keys():
            cell = came_from[cell]
            total_path.insert(0, cell)


        total_path = np.array(total_path)
        traj_world_x, traj_world_y = self.grid.conv_map_to_world(total_path[:, 0], total_path[:, 1])
        return np.vstack((traj_world_x, traj_world_y))


    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """

        start: Tuple[int, int] = self.grid.conv_world_to_map(start[0], start[1])
        goal: Tuple[int, int] = self.grid.conv_world_to_map(goal[0], goal[1])

        # creates a copy of occupancy map to modify it and take into account
        # a margin in the walls
        self.map_walls = copy.deepcopy(self.grid.occupancy_map)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
        self.map_walls = cv2.dilate(self.map_walls, kernel, iterations=1)

        #cv2.imshow("map_walls", self.map_walls)

        # min heap to contain values to explore next
        open_set = [(0.0, start)]
        heapq.heapify(open_set)

        # dictionary to trace back route
        came_from = {}

        # cost to get to each cell
        g_score = defaultdict(lambda: math.inf)
        g_score[start] = 0.0


        # best guess of cost for each cell (cost + heuristic)
        f_score = defaultdict(lambda: math.inf)
        f_score[start] = 0.0 + self.heuristic(start, goal)


        while len(open_set) > 0:
            current = heapq.heappop(open_set)
            current_f, current_cell = current
            # lazy deletion: skip stale entries
            if current_f > f_score[current_cell]:
                continue
            if current_cell == goal:
                return self.reconstruct_path(came_from, goal)

            neighbours = self.get_neighbors(current_cell)
            for cell in neighbours:
                # if cell is a wall, skip it
                if self.map_walls[cell[0], cell[1]] > 0:
                     continue
                tentative_g_score = g_score[current_cell] + self.heuristic(current_cell, cell)
                if tentative_g_score < g_score[cell]:
                    # better path, recording it
                    came_from[cell] = current_cell
                    g_score[cell] = tentative_g_score
                    f_score[cell] = tentative_g_score + 5*self.heuristic(cell, goal)
                    heapq.heappush(open_set, (f_score[cell], cell))


        # goal was never reached
        print('failed getting to objective')
        return None


    # def explore_frontiers(self):
    #     """ Frontier based exploration """
    #     goal = np.array([0, 0, 0])  # frontier to reach for exploration
    #     return goal
