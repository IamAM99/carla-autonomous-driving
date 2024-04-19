from typing import Dict
import tensorflow as tf

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
INITIAL_WAITING_TIME: float = 2.0

# Training
MAX_EPISODES: int = 10
MAX_STEPS_PER_EPISODE: int = 10_000
NUM_GREEDY_FRAMES: int = 5_000 # Number of frames for exploration
MIN_REPLAY_MEMORY_LEN: int = 100 # Minimum replay memory length to start training
MAX_REPLAY_MEMORY_LEN: int = 5_000 # Maximum replay memory length. the Deepmind paper: 1_000_000
TARGET_MODEL_UPDATE_INTERVAL: int = 1_000 # Update the target model every ? actions
ACTIONS_PER_SECOND: int = 60 # number of actions per second when selecting randomly
TRAIN_PER_SECOND: int = 10 # number of model training per second

# Model
MODEL_TYPE: str = "mlp" # "mlp" or "cnn"
GAMMA: float = 0.99
EPSILON: float = 1.0
EPSILON_MIN: float = 0.1
EPSILON_MAX: float = 1.0
BATCH_SIZE: int = 32
OPTIMIZER_FUNC = tf.keras.optimizers.Adam
LEARNING_RATE: float = 0.01
NUM_WAYPOINT_FEATURES: int = 6 # number of waypoints to pass into the feature list