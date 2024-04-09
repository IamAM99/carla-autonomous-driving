from typing import Dict

# Client
HOST_IP: str = "127.0.0.1"
PORT: int = 2000

# Action
STEER_AMT: float = 0.8
ACTIONS: Dict[int, Dict[str, float]] = {
    0: dict(throttle=1, steer=0*STEER_AMT, brake=0), # forward
    1: dict(throttle=0, steer=-1*STEER_AMT, brake=0), # steer left
    2: dict(throttle=0, steer=1*STEER_AMT, brake=0), # steer right
    3: dict(throttle=0, steer=0*STEER_AMT, brake=1), # brake
}

# Rewards
COLLISION_REWARD: float = -200.0
GOAL_DISTANCE_THRESHOLD: float = 1.0 # minimum distance of the car from the goal point to count as success
GOAL_REACHED_REWARD: float = 100.0

# Camera
IM_WIDTH: int = 480
IM_HEIGHT: int = 270
FOV: int = 110
SHOW_CAM: bool = True
SAVE_IMG: bool = False
LOCK_SPECTATOR_VIEW: bool = False

# Simulation
NO_RENDERING_MODE: bool = False
SECONDS_PER_EPISODE: float = 10.0