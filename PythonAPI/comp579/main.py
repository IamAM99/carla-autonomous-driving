import glob
import os
import sys

import time
import numpy as np

from environment import CarlaEnv

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

sys.path.append("../carla")

import carla
from agents.tools.misc import draw_locations

np.set_printoptions(precision=3, suppress=True)

def main():
    env = CarlaEnv()
    draw_locations(env.world, env.route_points[:-1])
    draw_locations(env.world, [env.route_points[-1]], color=carla.Color(0, 128, 0))

    try:
        env.reset()
        
        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}, distance = {states['d']:.2f}")
        print(f"Waypoints: \n{states['waypoints']}")
        time.sleep(2)

        states, reward, _, _ = env.step(1)
        print(f"R = {reward:.2f}, distance = {states['d']:.2f}")
        print(f"Waypoints: \n{states['waypoints']}")
        time.sleep(1)

        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}, distance = {states['d']:.2f}")
        print(f"Waypoints: \n{states['waypoints']}")
        time.sleep(1)

        states, reward, _, _ = env.step(2)
        print(f"R = {reward:.2f}, distance = {states['d']:.2f}")
        print(f"Waypoints: \n{states['waypoints']}")
        time.sleep(1)

        states, reward, _, _ = env.step(0)
        print(f"R = {reward:.2f}, distance = {states['d']:.2f}")
        print(f"Waypoints: \n{states['waypoints']}")
        time.sleep(2)

    finally:
        env.clear()
        print("Cleared successfully")

if __name__=="__main__":
    main()
