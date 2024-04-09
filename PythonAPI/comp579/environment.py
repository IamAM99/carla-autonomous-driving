import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import config as cfg
from typing import Tuple, List

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("../carla")

import carla
from agents.tools.misc import draw_waypoints


class CarlaEnv:
    def __init__(self, host: str = cfg.HOST_IP, port: int = cfg.PORT, *args, **kwargs):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)

        # world and map attributes
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spectator = self.world.get_spectator()
        self.spawn_points = self.map.get_spawn_points()
        self.bp_lib = self.world.get_blueprint_library()

        # the car blueprint
        self.model_3 = self.bp_lib.find("vehicle.tesla.model3")
        
        self.collision_hist: list = []
        self.actor_list: list = []
        self.car_spawn_point: carla.Transform = None
        self.vehicle: carla.Actor = None
        self.camera: carla.Sensor = None
        self.collision_sensor: carla.Sensor = None
        self.episode_start: float = time.time()
        self.front_camera: np.typing.ArrayLike = None
        self.loc: carla.Transform = None
        self.waypoint: carla.Waypoint = None
        self.waypoints: np.typing.ArrayLike = None # [x_column; y_column]

    def reset(self, car_spawn_point=None, *args, **kwargs):
        if car_spawn_point is None:
            car_spawn_point = self.spawn_points[0]
        self.car_spawn_point = car_spawn_point
        self.collision_hist = []
        self.actor_list = []

        self._spawn_vehicle()
        self._spawn_camera()

        # an initial control needs to be here for carla to accept next control commands without delay
        # after a wait time, we disengage the brakes
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        
        # wait for the car to fall and settle
        time.sleep(4)

        # wait for the camera to start capturing
        while self.front_camera is None:
            time.sleep(0.01)

        # set up collision sensor (after the vehicle settled)
        self._spawn_col_sensor()

        # set starting time
        self.episode_start = time.time()

        # disengage the brakes
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        return self.front_camera
    
    def step(self, action: int):
        # take action
        control = cfg.ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(**control))
        
        # get the states
        self._update_loc_waypoint()
        self._update_waypoints()
        distance, phi, v_kmh = self._get_states()

        # check if done and get the reward value
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        else:
            done = False
            reward = v_kmh * (np.abs(np.cos(phi)) - np.abs(np.sin(phi)) - distance)

        if self.episode_start + cfg.SECONDS_PER_EPISODE < time.time():
            done = True

        # create the states dictionary
        states = {
            "image": self.front_camera,
            "waypoints": self.waypoints,
            "d": distance,
            "phi": phi,
            "v_kmh": v_kmh,
        }

        return states, reward, done, None

    def clear(self):
        for actor in self.actor_list:
            # stop the actor's callback
            if hasattr(actor, 'is_listening') and actor.is_listening:
                actor.stop()

            # destroy the actor
            if actor.is_alive:
                actor.destroy()

        self.actor_list = []

    def _get_states(self) -> Tuple[float, float, float]:
        # distance from closest waypoint
        distance = self._calc_distance(self.loc.location, self.waypoint.transform.location)
        
        # angle difference with the closest waypoint
        phi = np.deg2rad(self.waypoint.transform.rotation.yaw - self.loc.rotation.yaw)

        # velocity
        v_vector = self.vehicle.get_velocity()
        v_kmh = 3.6 * np.sqrt(v_vector.x**2 + v_vector.y**2 + v_vector.z**2)
        

        return distance, phi, v_kmh

    def _update_loc_waypoint(self) -> Tuple[carla.Transform, carla.Waypoint]:
        self.loc = self.vehicle.get_transform()
        self.waypoint = self.map.get_waypoint(self.loc.location)

        return self.loc, self.waypoint
    
    def _update_waypoints(self) -> List[carla.Waypoint]:
        self.waypoints = [self.waypoint]
        while len(self.waypoints) < 15:
            self.waypoints += self.waypoints[-1].next(10)
        
        draw_waypoints(self.world, self.waypoints)
        self.waypoints = np.array([[w.transform.location.x, w.transform.location.y] for w in self.waypoints[:15]])

        return self.waypoints

    def _spawn_vehicle(self):
        if self.car_spawn_point is None:
            self.car_spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(self.model_3, self.car_spawn_point)
        
        self.actor_list.append(self.vehicle)

        self._update_loc_waypoint()
        self._update_waypoints()

        return self.vehicle
    
    def _spawn_camera(self):
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{cfg.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{cfg.IM_HEIGHT}")
        cam_bp.set_attribute("fov", f"{cfg.FOV}")

        self.camera = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)

        self.actor_list.append(self.camera)

        self.camera.listen(lambda data: self._process_img(data)) # start capturing the image
 
        return self.camera

    def _spawn_col_sensor(self):
        colsensor = self.bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(colsensor, transform=carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)
        self.collision_sensor.listen(lambda event: self._collision_data(event))

        return self.collision_sensor

    def _process_img(self, data):
        self.front_camera = np.array(data.raw_data).reshape((cfg.IM_HEIGHT, cfg.IM_WIDTH, 4))[:, :, :3]

        if cfg.LOCK_SPECTATOR_VIEW:
            self.spectator.set_transform(get_transform(self.vehicle.get_location()))
        
        if cfg.SHOW_CAM:
            cv2.imshow("", self.front_camera)
            cv2.waitKey(1)
        
        if cfg.SAVE_IMG:
            data.save_to_disk('_out/%06d.png' % data.frame_number)

    def _calc_distance(self, loc1: carla.Location, loc2: carla.Location):
        dist_obj = loc1 - loc2
        dist_arr = np.array([dist_obj.x, dist_obj.y, dist_obj.z])

        return np.sqrt(np.sum(np.square(dist_arr)))
     
    def _collision_data(self, event):
        name = ' '.join(event.other_actor.type_id.replace('_', '.').title().split('.')[1:])
        truncate = 250
        print((name[:truncate - 1] + u'\u2026') if len(name) > truncate else name)

        self.collision_hist.append(event)
    
def get_transform(vehicle_location, angle=-90, d=3):
    """ Get the transform for the spectator view
    """
    a = np.radians(angle)
    location = carla.Location(d * np.cos(a), d * np.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-30))