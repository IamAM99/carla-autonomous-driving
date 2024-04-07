# Client
HOST_IP = 127.0.0.1
PORT = 2000

# Action
STEER_AMT = 0.8
ACTIONS = {
    0: dict(throttle=1, steer=0*STEER_AMT, brake=0), # forward
    1: dict(throttle=0, steer=-1*STEER_AMT, brake=0), # steer left
    2: dict(throttle=0, steer=1*STEER_AMT, brake=0), # steer right
    3: dict(throttle=0, steer=0*STEER_AMT, brake=1), # brake
}

# Camera
IM_WIDTH = 480
IM_HEIGHT = 270
FOV = 110
SHOW_CAM = True

# Simulation
SECONDS_PER_EPISODE = 10