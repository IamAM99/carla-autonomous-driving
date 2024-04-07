import glob
import os
import random
import sys
import time

import cv2
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


class CarlaEnv:
    IM_WIDTH = 640
    IM_HEIGHT = 480
    FOV = 110
    SHOW_CAM = True
    SECONDS_PER_EPISODE = 10
    STEER_AMT = 0.5

    def __init__(self, host="127.0.0.1", port=2000, *args, **kwargs):
        self.client = carla.Client(host, port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.bp_lib = self.world.get_blueprint_library()
        self.model_3 = self.bp_lib.find("vehicle.tesla.model3")
        
        self.collision_hist = []
        self.actor_list = []
        self.spawn_point = None
        self.vehicle = None
        self.sensor = None
        self.episode_start = 0
        self.front_camera = None


    def reset(self, car_spawn_point=None, *args, **kwargs):
        self.spawn_point = car_spawn_point
        self.collision_hist = []
        self.actor_list = []

        self._spawn_vehicle()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        self._spawn_camera()
        self.sensor.listen(lambda data: self._process_img(data))

        time.sleep(2)
        while self.front_camera is None:
            time.sleep(0.01)

        # collsensor = self.bp_lib.find("sensor.other.collision")
        # self.colsensor = self.world.spawn_actor(collsensor, attach_to=self.vehicle)
        # self.actor_list.append(self.colsensor)
        # self.colsensor.listen(lambda event: self._collision_data(event))

        self.episode_start = time.time()
        # self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.front_camera
    
    def step(self, action):
        if action==0:
            control = (1.0, -1*self.STEER_AMT)
        elif action==1:
            control = (1.0, 0)
        elif action==2:
            control = (1.0, 1*self.STEER_AMT)

        self.vehicle.apply_control(carla.VehicleControl(*control))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
        
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
        elif kmh < 50:
            done = False
            reward = -1
        else:
            done = False
            reward = 1

        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            done = True
        
        reward = self._get_states()

        return self.front_camera, reward, done, None
    

    def clear(self):
        for actor in self.actor_list:
            actor.destroy()

    def _get_states(self):       
        # distance from closest waypoint
        loc = self.vehicle.get_transform()
        waypoint = self.map.get_waypoint(loc.location).transform
        distance = self._calc_distance(loc.location, waypoint.location)
        
        # angle difference with the closest waypoint
        phi = np.deg2rad(waypoint.rotation.yaw - loc.rotation.yaw)
        cos_phi = np.cos(phi)

        # velocity
        v_vector = self.vehicle.get_velocity()
        v_kmh = int(3.6 * np.sqrt(v_vector.x**2 + v_vector.y**2 + v_vector.z**2))
        

        return distance, cos_phi, v_kmh

    def _spawn_vehicle(self):
        if self.spawn_point is None:
            self.spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
        
        self.actor_list.append(self.vehicle)

        return self.vehicle
    
    def _spawn_camera(self):

        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        cam_bp = self.bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        cam_bp.set_attribute("fov", f"{self.FOV}")

        self.sensor = self.world.spawn_actor(cam_bp, spawn_point, attach_to=self.vehicle)

        self.actor_list.append(self.sensor)
        
        return self.sensor

    def _process_img(self, data):
        self.front_camera = np.array(data.raw_data).reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))[:, :, :3]
        
        if self.SHOW_CAM:
            cv2.imshow("", self.front_camera)
            cv2.waitKey(1)

    def _calc_distance(self, loc1: carla.Location, loc2: carla.Location):
        dist_obj = loc1 - loc2
        dist_arr = np.array([dist_obj.x, dist_obj.y, dist_obj.z])

        return np.sqrt(np.sum(np.square(dist_arr)))
        

    def _collision_data(self, event):
        self.collision_hist.append(event)
    

class Car:
    def __init__(self, *args, **kwargs):
        pass