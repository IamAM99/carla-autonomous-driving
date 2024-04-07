import time

import cv2
import numpy as np
from environment import CarlaEnv

try:
    import glob
    import os
    import sys
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

IM_WIDTH = 640
IM_HEIGHT = 480
CAM_FOV = 110


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]

    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0


env = CarlaEnv()
env.reset()

try:
    
    _, reward, _, _ = env.step(1)
    print(reward)

    time.sleep(5)

    _, reward, _, _ = env.step(2)
    print(reward)
finally:
    env.clear()

# try:
#     vehicle = client.spawn_vehicle()
#     sensor = client.spawn_camera(vehicle)

#     vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

#     sensor.listen(lambda data: process_img(data))

#     time.sleep(4)
# finally:
#     client.clear()