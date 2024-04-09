import glob
import os
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from typing import List

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla


def calc_distance(tr1: carla.Transform, tr2: carla.Transform) -> float:
    x1 = tr1.location.x
    y1 = tr1.location.y
    z1 = tr1.location.z

    x2 = tr2.location.x
    y2 = tr2.location.y
    z2 = tr2.location.z

    distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
    
    return distance


def get_transforms(filename="route.pickle", point_distance=8, show_plot=False) -> List[carla.Transform]:
    with open("route.pickle", 'rb') as handle:
        route_vars = pickle.load(handle)
    
    
    transform_args = [
        {
            "location": carla.Location(**args["Location"]),
            "rotation": carla.Rotation(**args["Rotation"]),
        }
        for args in route_vars
    ]

    all_transforms = [carla.Transform(**args) for args in transform_args]

    transforms = [all_transforms[0]]
    for transform in all_transforms[1:]:
        if calc_distance(transforms[-1], transform) > point_distance:
            transforms.append(transform)

    if show_plot:
        x = [transform.location.x for transform in transforms]
        y = [transform.location.y for transform in transforms]

        plt.plot(x, y, '.')
        plt.xlim([50, 250])
        plt.ylim([0, 200])
        plt.show()

    return transforms


def main():
    tr = get_transforms(show_plot=True)
    print(len(tr))


if __name__=="__main__":
    main()