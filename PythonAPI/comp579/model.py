import os 

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle 
from datetime import datetime

import config as cfg


class DQNAgent:
    def __init__(self, env):
        self.env = env 
        self.epsilon = cfg.EPSILON
        self.optimizer = cfg.OPTIMIZER_FUNC(learning_rate=cfg.LEARNING_RATE)
        self.num_actions = len(cfg.ACTIONS)
    
    def update(self,):
        pass
    
    def train(self,):
        model = self._create_q_model()
        model_target = self._create_q_model()
        # Experiment history
        action_history = []
        state_history = []
        state_next_history = []
        reward_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0
        
        while True:
            # resetting the environment
            state = self.env.reset()
            state = self._reshape_input(state)
            
            episode_reward = 0
            
            for timestep in range(1, cfg.MAX_STEPS_PER_EPISODE):
                frame_count += 1

                # taking epsilon-greedy action
                if frame_count < cfg.NUM_RANDOM_FRAMES or cfg.EPSILON > np.random.rand(1)[0]:
                    action = np.random.choice(self.num_actions)
                else:
                    # taking actions based on Q-value estimations
                    action_probs = model(np.expand_dims(state, 0), training=False)
                    # take the best action
                    action_probs = action_probs.numpy()
                    action = action_probs.argmax(axis=1).item()
                    
                # decay the epsilon
                self.epsilon -= (cfg.EPSILON_MAX - cfg.EPSILON_MIN) / cfg.NUM_GREEDY_FRAMES
                self.epsilon = max(self.epsilon, cfg.EPSILON_MIN)
                
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

                # Update every fourth frame and once batch size is over 32
                if frame_count % cfg.MODEL_COOLDOWN_FRAMES == 0 and len(done_history) > cfg.BATCH_SIZE:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=cfg.BATCH_SIZE)

                    # Using list comprehension to sample from replay buffer
                    state_sample = np.array([state_history[i] for i in indices])
                    state_next_sample = np.array([state_next_history[i] for i in indices])
                    rewards_sample = [reward_history[i] for i in indices]
                    action_sample = [action_history[i] for i in indices]
                    done_sample = tf.convert_to_tensor(
                        [float(done_history[i]) for i in indices]
                    )

                    # Build the updated Q-values for the sampled future states
                    # Use the target model for stability
                    future_rewards = model_target.predict(state_next_sample)
                    # Q value = reward + discount factor * expected future reward
                    updated_q_values = rewards_sample + cfg.GAMMA * tf.reduce_max(future_rewards, axis=1)

                    # If final frame set the last value to -1
                    updated_q_values = updated_q_values * (1 - done_sample) - done_sample

                    # Create a mask so we only calculate loss on the updated Q-values
                    masks = tf.one_hot(action_sample, self.num_actions)

                    with tf.GradientTape() as tape:
                        # Train the model on the states and updated Q-values
                        q_values = model(state_sample)

                        # Apply the masks to the Q-values to get the Q-value for action taken
                        q_action = tf.reduce_sum(tf.math.multiply(q_values, masks), axis=1)
                        # Calculate loss between new Q-value and old Q-value
                        # print(f"updated q: {updated_q_values}")
                        # print(f"q: {q_action}")
                        loss = keras.losses.MSE(updated_q_values, q_action)
                    # Backpropagation
                    grads = tape.gradient(loss, model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
                if frame_count % cfg.TARGET_MODEL_COOLDOWN_FRAMES == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(reward_history) > cfg.MAX_REPLAY_MEMORY_LEN:
                    del reward_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
                    print(f"Episode {episode_count} done at frame {frame_count}")
                    break

            # Update running reward to check condition for solving
            episode_reward_history.append(episode_reward)
            if len(episode_reward_history) > 100:
                del episode_reward_history[:1]
            running_reward = np.mean(episode_reward_history)

            episode_count += 1

            if running_reward > 4000:  # Condition to consider the task solved
                print("Solved at episode {}!".format(episode_count))
                break

            if (
                cfg.MAX_EPISODES > 0 and episode_count >= cfg.MAX_EPISODES
            ):  # Maximum number of episodes reached
                print("Stopped at episode {}!".format(episode_count))
                break
        
        # saving the history of the experiment
        experiment_name = f"DQN_{cfg.MODEL_TYPE}_reward_{running_reward}" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Saving {experiment_name}")
        
        history = {
            "action": action_history,
            "reward": reward_history,
            "state": state_history,
            "done": done_history,
        }

        with open("../artifacts/history/"+experiment_name+"_history.json", "w") as history_file:
            pickle.dump(history, history_file)
        
        model_target.save("../artifacts/models/"+experiment_name+"_model.keras")
        
        
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
        frame_count = 0
        
        # resetting the environment
        state = self.env.reset()
        state = self._reshape_input(state)
            
        
        for timestep in range(1, cfg.MAX_STEPS_PER_EPISODE):
            frame_count += 1
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
                print(f"Episode is done at frame {frame_count}")
                break
        
        
        history = {
            "action": action_history,
            "reward": reward_history,
            "state": state_history,
            "done": done_history,
        }
        
        return history, episode_reward
        
        
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
        
        return state