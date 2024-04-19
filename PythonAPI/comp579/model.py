import os 

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # don't print warning and infor messages

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import json 
import random
import time
from threading import Thread
from datetime import datetime
from collections import deque
from tqdm.auto import tqdm

import config as cfg


class DQNAgent:
    def __init__(self, env):
        self.env = env 
        self.epsilon = cfg.EPSILON
        self.optimizer = cfg.OPTIMIZER_FUNC(learning_rate=cfg.LEARNING_RATE)
        self.num_actions = len(cfg.ACTIONS)
        self.replay_memory = deque(maxlen=cfg.MAX_REPLAY_MEMORY_LEN)
        self.loss_func = keras.losses.MSE

        self.model = self._create_q_model()
        self.target_model = self._create_q_model()
        self.training_initialized = False
        self.terminate = False
        self.frame_count = 0

        self.trainer_thread = None
        self.loss_list = []

    def train(self,):
        # start training thread
        self.trainer_thread = Thread(target=self._training_loop, daemon=True)
        self.trainer_thread.start()
        while not self.training_initialized:
            time.sleep(0.1)

        # Experiment history
        history = {
            "state": [],
            "action": [],
            "reward": [],
            "episode_reward": [],
            "loss": [],
        }
        
        for episode in tqdm(range(1, cfg.MAX_EPISODES+1), position=0, leave=True, ascii=True, unit="episodes"):
            state = self.env.reset()
            state = self._reshape_input(state)

            episode_reward = 0

            while True:
                self.frame_count += 1

                # take an epsilon-greedy action
                action = self._take_action(state, is_random=(self.frame_count<cfg.NUM_RANDOM_FRAMES))

                # decay the epsilon
                self._decay_epsilon()

                # pass the action to the environment
                next_state, reward, done, _ = self.env.step(action)
                next_state = self._reshape_input(next_state)
                self._update_replay_memory((state, action, reward, next_state, done))

                # update logging variables
                history["state"].append(np.squeeze(state).tolist())
                history["action"].append(action)
                history["reward"].append(reward)
                episode_reward += reward

                # update the state
                state = next_state

                if done:
                    # print(f"Episode {episode} done at frame {self.frame_count}")
                    break
            
            history["episode_reward"].append(episode_reward)

        history["loss"] = self.loss_list

        # Maximum number of episodes reached
        print(f"Stopped at episode {episode}, frame {self.frame_count}.")
        
        # save the history of the experiment
        experiment_name = f"DQN_{cfg.MODEL_TYPE}_reward_{episode_reward}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Saving {experiment_name}")

        with open("../artifacts/history/"+experiment_name+"_history.json", "w") as history_file:
            json.dump(history, history_file)
        
        self.target_model.save("../artifacts/models/"+experiment_name+"_model.keras")

    def kill_thread(self,):
        self.terminate = True
        self.trainer_thread.join()

    def _update_replay_memory(self, transition):
        # transition: (current_state, action, reward, next_state, done)
        self.replay_memory.append(transition)

    def _take_action(self, state, is_random):
        if is_random or cfg.EPSILON > np.random.rand(1)[0]:
            action = np.random.choice(self.num_actions)
            time.sleep(1/20)
        else:
            # taking actions based on Q-value estimations
            action_probs = self.model(state, training=False)
            # take the best action
            action_probs = action_probs.numpy()
            action = action_probs.argmax(axis=1).item()
        
        return action

    def _decay_epsilon(self,):
        self.epsilon -= (cfg.EPSILON_MAX - cfg.EPSILON_MIN) / cfg.NUM_GREEDY_FRAMES
        self.epsilon = max(self.epsilon, cfg.EPSILON_MIN)
    
    def _fit_batch(self):
        """fit the model on the replay memory in a batch 
        """
        if len(self.replay_memory) < cfg.MIN_REPLAY_MEMORY_LEN:
            return False
        
        minibatch = random.sample(self.replay_memory, cfg.BATCH_SIZE)

        # unpack samples
        current_states = np.array([transition[0] for transition in minibatch])
        actions_list = np.array([transition[1] for transition in minibatch])
        rewards_list = np.array([transition[2] for transition in minibatch])
        future_states = np.array([transition[3] for transition in minibatch])
        done_list = tf.convert_to_tensor([float(transition[4]) for transition in minibatch])

        # predict Q-value using the target model
        future_q_list = self.target_model(future_states)
        future_q_list = tf.squeeze(future_q_list)

        # Q value = reward + discount factor * expected future reward
        updated_q_values = rewards_list + (1 - done_list) * cfg.GAMMA * tf.reduce_max(future_q_list, axis=1)

        # Create a mask so we only calculate loss on the updated Q-values
        masks = tf.one_hot(actions_list, self.num_actions)

        # train the model
        self._fit(current_states, updated_q_values, masks)

    def _fit(self, X, y, masks):
        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.model(X)
            q_values = tf.squeeze(q_values)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.math.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_func(y, q_action)

        # Backpropagation
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # log loss value
        self.loss_list.append(loss.numpy().item())

    def _update_target_model(self,):
        if self.frame_count % cfg.TARGET_MODEL_UPDATE_INTERVAL == 0:
            self.target_model.set_weights(self.model.get_weights())

    def _training_loop(self,):
        # call fit once to initialize tensorflow (to avoid future delays)
        random_states = tf.random.uniform(shape=(1, cfg.NUM_WAYPOINT_FEATURES+3))
        random_q = tf.random.uniform(shape=(1, self.num_actions))
        random_mask = tf.one_hot([0], self.num_actions)
        self._fit(random_states, random_q, random_mask)
        self.training_initialized = True

        # start the training process
        while True:
            if self.terminate:
                return
            self._fit_batch()
            self._update_target_model()
            time.sleep(0.01)
        
    def _create_q_model(self):
        if cfg.MODEL_TYPE == "cnn":
            return keras.Sequential(
                [
                    # Convolutions on the frames on the screen
                    layers.Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]),
                        output_shape=(cfg.IM_HEIGHT, cfg.IM_WIDTH, 4),
                        input_shape=(4, cfg.IM_HEIGHT, cfg.IM_WIDTH),
                    ),
                    layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(4, cfg.IM_HEIGHT, cfg.IM_WIDTH)),
                    layers.Conv2D(64, 4, strides=2, activation="relu"),
                    layers.Conv2D(64, 3, strides=1, activation="relu"),
                    layers.Flatten(),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(self.num_actions, activation="linear"),
                ]
            )
        elif cfg.MODEL_TYPE == "mlp":
            return keras.Sequential(
                [
                    layers.Dense(100, activation="relu", input_shape=(cfg.NUM_WAYPOINT_FEATURES+3,)),
                    layers.Dense(200, activation="relu"),
                    layers.Dense(self.num_actions, activation="linear"),
                ]
            )
        
        else:
            raise ValueError('Model type should be "mlp" or "cnn".')

    def _reshape_input(self, state):
        if cfg.MODEL_TYPE == "mlp":
            state = np.concatenate((state["waypoints"][:, 0],
                                    np.array([state["d"]]),
                                    np.array([state["phi"]]),
                                    np.array([state["v_kmh"]])))
        if cfg.MODEL_TYPE == "cnn":
            state = state["image"]
        
        return np.expand_dims(state, 0)

    def test(self, model_path):
        # loading mode
        model = tf.keras.models.load_model(model_path)
        
        # Experiment history
        action_history = []
        state_history = []
        state_next_history = []
        reward_history = []
        done_history = []
        episode_reward = 0
        
        # resetting the environment
        state = self.env.reset()
        state = self._reshape_input(state)
            
        
        for timestep in range(1, cfg.MAX_STEPS_PER_EPISODE):
            self.frame_count += 1
            # taking actions based on Q-value estimations
            action_probs = model(np.expand_dims(state, 0), training=False)
            # take the best action
            action_probs = action_probs.numpy()
            action = action_probs.argmax(axis=1).item()
            
            # taking the action
            next_state, reward, done, _ = self.env.step(action)
            next_state = self._reshape_input(next_state)
            episode_reward += reward

            # saving the actions, rewards and states
            action_history.append(action)
            state_history.append(state.tolist())
            state_next_history.append(next_state.tolist())
            done_history.append(done)
            reward_history.append(reward)
        
            # update the state
            state = next_state       
            
            if done:
                print(f"Episode is done at frame {self.frame_count}")
                break
        
        
        history = {
            "action": action_history,
            "reward": reward_history,
            "state": state_history,
            "done": done_history,
        }
        
        return history, episode_reward
