import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import config as cfg
from typing import Tuple, List

from reward_functions import RouteReward
import route

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
        # make a connection to the server
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        # control parameters
        self.num_waypoints = cfg.NUM_WAYPOINT_FEATURES

        # get the route transforms
        self.route_points: List[carla.Transform] = route.get_transforms(point_distance=4)

        # create a reward object
        self.reward = RouteReward(self.route_points)

        # world attributes
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.bp_lib = self.world.get_blueprint_library()

        # set default spectator view
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=-100, y=105, z=20), 
                carla.Rotation(pitch=-40, yaw=135, roll=0)
            )
        )
        
        # map attributes
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()

        # set weather
        self.world.set_weather(carla.WeatherParameters.CloudyNoon)

        # set the rendering mode
        settings = self.world.get_settings()
        if cfg.NO_RENDERING_MODE:
            settings.no_rendering_mode = True
            self.world.apply_settings(settings)
        else:
            settings.no_rendering_mode = False
            self.world.apply_settings(settings)

        # the vehicle
        self.model_3 = self.bp_lib.find("vehicle.tesla.model3")
        self.car_spawn_point: carla.Transform = None
        self.vehicle: carla.Actor = None
        self.vehicle_transform: carla.Transform = None
        self.waypoint: carla.Waypoint = None
        self.waypoints: np.typing.ArrayLike = None # [x_column; y_column; z_column]
        
        # sensors
        self.camera: carla.Sensor = None
        self.show_cam: bool = cfg.SHOW_CAM
        self.front_camera: np.typing.ArrayLike = None # output of the front camera
        self.collision_sensor: carla.Sensor = None
        self.collision_hist: list = []
        self.lane_sensor: carla.Sensor = None
        self.crossed_center_line: bool = False

        # initialization of other attributes
        self.episode_start: float = time.time()
        self.sensor_list: list = []

    def reset(self, car_spawn_point: carla.Location = None, *args, **kwargs):
        self.clear()

        if car_spawn_point is None:
            car_spawn_point = self.spawn_points[101]
        self.car_spawn_point = car_spawn_point

        self._spawn_vehicle()
        self._spawn_camera()

        # an initial control needs to be here for carla to accept next control commands without delay
        # after a wait time, we disengage the brakes
        self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, brake=1.0))
        
        # wait for the car to fall and settle
        time.sleep(cfg.INITIAL_WAITING_TIME)
        
        # disengage the brakes
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # wait for the camera to start capturing
        while self.front_camera is None:
            time.sleep(0.01)

        # set up other sensors (after the vehicle settled)
        self._spawn_col_sensor()
        self._spawn_lane_sensor()

        # set starting time
        self.episode_start = time.time()


        # get the states
        self._update_loc_waypoint()
        self._update_waypoints()
        _, distance, phi, v_kmh = self.reward.get(
            self.vehicle_transform, 
            self.waypoint.transform, 
            self.vehicle.get_velocity()
        )

        states = {
            "image": self.front_camera,
            "waypoints": self.waypoints,
            "d": distance,
            "phi": phi,
            "v_kmh": v_kmh,
        }
        
        return states
    
    def step(self, action: int):
        # take action
        control = cfg.ACTIONS[action]
        self.vehicle.apply_control(carla.VehicleControl(**control))
        
        # get the states
        self._update_loc_waypoint()
        self._update_waypoints()
        reward, distance, phi, v_kmh = self.reward.get(
            self.vehicle_transform, 
            self.waypoint.transform, 
            self.vehicle.get_velocity()
        )

        # check if done and get the reward value
        if (len(self.collision_hist) != 0) or self.crossed_center_line or distance > 4:
            is_done = True
            reward = cfg.COLLISION_REWARD
        elif self._calc_distance(self.vehicle_transform.location, self.route_points[-1].location) < cfg.GOAL_DISTANCE_THRESHOLD:
            is_done = True
            reward = cfg.GOAL_REACHED_REWARD
        else:
            is_done = False

        # end the training if time limit is reached
        if self.episode_start + cfg.SECONDS_PER_EPISODE < time.time():
            is_done = True

        # create the states dictionary
        states = {
            "image": self.front_camera,
            "waypoints": self.waypoints,
            "d": distance,
            "phi": phi,
            "v_kmh": v_kmh,
        }

        return states, reward, is_done, None

    def clear(self, final=False):
        self.show_cam = False
        if final:
            self.client.set_timeout(30.0)
            all_actors = self.world.get_actors()
            sensor_list = all_actors.filter("*sensor*")
            vehicle_list = all_actors.filter("*vehicle*")
        else:
            sensor_list = self.sensor_list
            vehicle_list = [self.vehicle]

        for sensor in sensor_list:
            if sensor.is_listening:
                sensor.stop()
            sensor.destroy()
        
        for vehicle in vehicle_list:
            if vehicle:
                vehicle.destroy()
        self.show_cam = cfg.SHOW_CAM
        self.sensor_list = []
        self.waypoint = None
        self.waypoints = None
        self.vehicle = None
        self.vehicle_transform = None
        self.camera = None
        self.front_camera = None
        self.collision_sensor = None
        self.collision_hist = []
        self.lane_sensor = None
        self.crossed_center_line = False
        
    def _update_loc_waypoint(self) -> Tuple[carla.Transform, carla.Waypoint]:
        self.vehicle_transform = self.vehicle.get_transform()
        self.waypoint = self.map.get_waypoint(self.vehicle_transform.location)

        return self.vehicle_transform, self.waypoint
    
    def _update_waypoints(self) -> List[carla.Waypoint]:
        self.waypoints = [self.waypoint]
        while len(self.waypoints) < self.num_waypoints:
            self.waypoints += self.waypoints[-1].next(10)
        
        # show the waypoints in the game
        draw_waypoints(self.world, self.waypoints[:self.num_waypoints])

        # yaw angle and the location coordinates of the vehicle
        phi_c = self.vehicle_transform.rotation.yaw
        loc_c = self.vehicle_transform.location

        # rotation matrix which rotates with respect to the vehicle's facing direction
        rotation_matrix = np.array(
            [
                [np.cos(phi_c), np.sin(phi_c), 0],
                [-np.sin(phi_c), np.cos(phi_c), 0],
                [0, 0, 1],
            ]
        )

        # apply the translation and rotation on the waypoints
        transformed_waypoints = []
        for waypoint in self.waypoints[:self.num_waypoints]:
            w_loc = waypoint.transform.location - loc_c
            new_w_loc = rotation_matrix @ np.array([w_loc.x, w_loc.y, w_loc.z]).T
            transformed_waypoints.append(new_w_loc)

        self.waypoints = np.array(transformed_waypoints)
        
        return self.waypoints

    def _spawn_vehicle(self):
        if self.car_spawn_point is None:
            self.car_spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(self.model_3, self.car_spawn_point)
        
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

        self.camera.listen(lambda data: self._process_img(data)) # start capturing the image
        
        self.sensor_list.append(self.camera)
 
        return self.camera

    def _spawn_col_sensor(self):
        col_sensor = self.bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(col_sensor, transform=carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._collision_data(event))
        self.sensor_list.append(self.collision_sensor)

        return self.collision_sensor
    
    def _spawn_lane_sensor(self):
        lane_sensor = self.bp_lib.find("sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(lane_sensor, transform=carla.Transform(), attach_to=self.vehicle)
        self.lane_sensor.listen(lambda event: self._lane_data(event))
        self.sensor_list.append(self.lane_sensor)

        return self.lane_sensor
    
    def _process_img(self, data):
        self.front_camera = np.array(data.raw_data).reshape((cfg.IM_HEIGHT, cfg.IM_WIDTH, 4))[:, :, :3]

        if cfg.LOCK_SPECTATOR_VIEW:
            self.spectator.set_transform(get_transform(self.vehicle.get_location()))
        
        if self.show_cam:
            cv2.imshow("front_camera", self.front_camera)
            cv2.waitKey(1)
        elif cv2.getWindowProperty("front_camera", cv2.WND_PROP_VISIBLE) > 0:
            cv2.destroyWindow("front_camera")
        
        if cfg.SAVE_IMG:
            data.save_to_disk('_out/%06d.png' % data.frame_number)

    def _calc_distance(self, loc1: carla.Location, loc2: carla.Location):
        dist_obj = loc1 - loc2
        dist_arr = np.array([dist_obj.x, dist_obj.y, dist_obj.z])

        return np.sqrt(np.sum(np.square(dist_arr)))
     
    def _collision_data(self, event):
        name = ' '.join(event.other_actor.type_id.replace('_', '.').title().split('.')[1:])
        truncate = 250
        # print((name[:truncate - 1] + u'\u2026') if len(name) > truncate else name)

        self.collision_hist.append(event)

    def _lane_data(self, event):
        lane_types = set(x.type for x in event.crossed_lane_markings)
        if (carla.libcarla.LaneMarkingType.SolidSolid in lane_types):# or\
        #    (carla.libcarla.LaneMarkingType.Solid in lane_types):
            self.crossed_center_line = True
    
def get_transform(vehicle_location, angle=-90, d=3):
    """ Get the transform for the spectator view
    """
    a = np.radians(angle)
    location = carla.Location(d * np.cos(a), d * np.sin(a), 2.0) + vehicle_location
    return carla.Transform(location, carla.Rotation(yaw=180 + angle, pitch=-30))