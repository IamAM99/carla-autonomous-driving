import glob
import os
import sys
import numpy as np
from typing import List

import config as cfg

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class RouteReward:
    def __init__(self, route_points: List[carla.Transform]):
        self.route_points = route_points
    
    def get(
        self, 
        vehicle_transform: carla.Transform, 
        waypoint_transform: carla.Transform,
        v_vector: carla.Vector3D,
        distance_threshold: float = cfg.MIN_WAYPOINT_DISTANCE_TO_RECORD,
    ) -> float:
        # minimum distance from the list of route points
        distances = [
            self._calc_manhattan_distance(point.location, vehicle_transform.location) 
            for point in self.route_points
        ]

        idx_min_dist = np.argmin(distances)
        
        distance = self._calc_euclidean_distance(
            self.route_points[idx_min_dist].location, 
            vehicle_transform.location,
        )

        distance = 0.0 if distance < distance_threshold else distance
        
        # angle difference with the closest waypoint
        phi = np.deg2rad(waypoint_transform.rotation.yaw - vehicle_transform.rotation.yaw)

        # velocity
        v_kmh = 3.6 * np.sqrt(v_vector.x**2 + v_vector.y**2 + v_vector.z**2)


        reward = v_kmh * (np.abs(np.cos(phi)) - np.abs(np.sin(phi)) - distance)

        return reward, distance, phi, v_kmh

    def _calc_manhattan_distance(self, l1: carla.Location, l2: carla.Location) -> float:
        distance = np.abs(l1.x-l2.x) + np.abs(l1.y-l2.y) + np.abs(l1.z-l2.z)
        return distance

    def _calc_euclidean_distance(self, l1: carla.Location, l2: carla.Location) -> float:
        distance = np.sqrt((l1.x-l2.x)**2 + (l1.y-l2.y)**2 + (l1.z-l2.z)**2)
        return distance