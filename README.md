# COMP 579: CARLA Project

## Instructions for Running the Code
Follow the steps below to run an experiment:
- Download and extract [CARLA 0.9.5](https://github.com/carla-simulator/carla/releases/tag/0.9.5). 
- Copy and replace the contents of this project into the CARLA folder. 
- Install the requirements (`pip install -r requirements.txt`). The python version should be 3.7.* (preferably 3.7.16).
- Run the `.\CarlaUE4.exe` (or `./CarlaUE4.sh` on linux) file.
- Change directory to `.\PythonAPI\comp579`.
- Modify `config.py` with the desired parameters. 
- Run `train.py` to train a DQN based on the set parameters. After the training is done, the model will be saved in `artifacts\models` and the history will be saved in `artifacts\history`. 

The history file contains the following keys:
  - `"state"`: list of states in each step.
  - `"action"`: action taken in each step.
  - `"reward"`: the reward received in each step.
  - `"episode_reward"`: the commulative reward (return) of each episode.
  - `"average_reward"`: the average reward values in each episode.
  - `"loss"`: the MSE loss of the model after each training.

Other executable scripts:
 - `my_manual_control`: used to control the car with "wasd" keys (requires `pygame`). The car is spawned at the start of the path, and the path is highlighted on the road. This script can be used to create a route and save it into `route.pickle`. To do so, you should pass the `--newroute` argument when running.
 - `test_env.py`: used to test the functionality of the environment. Spawns a car and performs some actions on it.

## Control Parameters
The following is a list of all control parameters that can be modified in `config.py`.
| Parameter                              | Description                                                                                                                   |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| HOST_IP: str                           | Server IP address                                                                                                             |
| PORT: int                              | Server port                                                                                                                   |
| STEER_AMT: float                       | Amount of steering [0, 1]                                                                                                     |
| ACTIONS: Dict[int, Dict[str, float]]   | Actions defined as a dictionary where keys are action number and the values are dictionaries with action specifications       |
| COLLISION_REWARD: float                | The amount of reward for collision or moving out of the road (either by passing the curb or the yellow line)                  |
| GOAL_DISTANCE_THRESHOLD: float         | How many meters the car should be  close to the destination to count as done [0, inf)                                         |
| GOAL_REACHED_REWARD: float             | The amount of reward for reaching the goal                                                                                    |
| REWARD_NORMALIZATION_FACTOR: float     | The factor by which the calculated reward  will be multiplied                                                                 |
| MIN_WAYPOINT_DISTANCE_TO_RECORD: float | Minimum distance that the car should get from the waypoints for the distance to be considered in the reward function [0, inf) |
| IM_WIDTH: int                          | Width of the front camera image [1, inf)                                                                                      |
| IM_HEIGHT: int                         | Height of the front camera image [1, inf)                                                                                     |
| FOV: int                               | Field of view of the front camera                                                                                             |
| SHOW_CAM: bool                         | Whether to show the front camera data in a  figure                                                                            |
| SAVE_IMG: bool                         | Whether to save the front camera image                                                                                        |
| LOCK_SPECTATOR_VIEW: bool              | Whether to reset the spectator view at  the start of each episode                                                             |
| NO_RENDERING_MODE: bool                | Whether the game engine should render the environment or not                                                                  |
| SECONDS_PER_EPISODE: float             | Maximum duration of each episode until the episode becomes done [0, inf)                                                      |
| INITIAL_WAITING_TIME: float            | The duration that we should wait for the car to settle down after spawning [0, inf)                                           |
| MAX_EPISODES: int                      | Number of experiment episodes [1, inf)                                                                                        |
| MAX_STEPS_PER_EPISODE: int             | Maximum number of steps per episode [1, inf)                                                                                  |
| NUM_GREEDY_FRAMES: int                 | Number of steps to take until epsilon reaches the EPSILON_MIN value [0, inf)                                                  |
| MIN_REPLAY_MEMORY_LEN: int             | The minimum size of the replay memory to start the training [1, inf)                                                          |
| MAX_REPLAY_MEMORY_LEN: int             | The maximum size of the replay memory (maximum  number) of previous steps to choose from for  training [1, inf)               |
| TARGET_MODEL_UPDATE_INTERVAL: int      | Number of steps to take before updating the target model [1, inf)                                                             |
| ACTIONS_PER_SECOND: int                | Maximum number of steps taken in each second (might be limited by the processing power) [1, inf)                              |
| TRAIN_PER_SECOND: int                  | Number of times per second to train the model [1, inf)                                                                        |
| MODEL_TYPE: str                        | Model structure. Either "mlp" or "cnn"                                                                                        |
| GAMMA: float                           | Amount of the discount factor [0, 1]                                                                                          |
| EPSILON: float                         | The starting value for epsilon [0, 1]                                                                                         |
| EPSILON_MIN: float                     | The minimum value of epsilon [0, 1]                                                                                           |
| EPSILON_MAX: float                     | The maximum value of epsilon [0, 1]                                                                                           |
| BATCH_SIZE: int                        | The number of previous steps to use for each training epoch [1, inf)                                                          |
| OPTIMIZER_FUNC: tf.keras.Optimizer     | Optimizer for model training                                                                                                  |
| LEARNING_RATE: float                   | Learning rate of the training [0, inf)                                                                                        |
| NUM_WAYPOINT_FEATURES: int             | Number of next waypoints in front of the car to use as features for training the model                                        |

## Other Modules and Scripts
- `environment.py`: contains the `CarlaEnv` class, which defines the environment. 
- `model.py`: contains the `DQNAgent` class which defines the model.
- `route.py`: post-processes the `route.pickle` file generated using `my_manual_control.py`.
- `reward_functions.py`: contains the `RouteReward` class which uses the waypoints in `route.pickle` to calculate the distance of the car from the closest waypoint and calculate the reward.
