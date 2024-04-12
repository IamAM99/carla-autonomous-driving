import os 

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf 
import keras 
from keras import layers
import numpy as np 

from environment import CarlaEnv

class DQNAgent:
    max_steps_per_episode = 10000
    max_episodes = 10
    IMG_HEIGHT = 84
    IMG_WIDTH = 84
    FEATURE_LENGTH = 8
    
    def __init__(self, gamma=0.99, epsilon=1.0, epsilon_min=0.1,
                 epsilon_max=1.0, batch_size=32, model_type="mlp",
                 optimizer=keras.optimizers.Adam, learning_rate=0.01, num_actions=4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = epsilon_min
        self.max_epsilon = epsilon_max
        self.epsilon_interval = epsilon_max - epsilon_min
        self.batch_size = batch_size 
        self.model_type = model_type  
        self.optimizer = optimizer(learning_rate=learning_rate)
        # self.loss_fcn = loss_fcn  
        self.num_actions = num_actions
    
    def create_q_model(self):
        if self.model_type == "cnn":
            return keras.Sequential(
                [
                    # Convolutions on the frames on the screen
                    layers.Lambda(lambda tensor: tf.transpose(tensor, [0, 2, 3, 1]),
                        output_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 4),
                        input_shape=(4, self.IMG_HEIGHT, self.IMG_WIDTH),
                    ),
                    layers.Conv2D(32, 8, strides=4, activation="relu", input_shape=(4, self.IMG_HEIGHT, self.IMG_WIDTH)),
                    layers.Conv2D(64, 4, strides=2, activation="relu"),
                    layers.Conv2D(64, 3, strides=1, activation="relu"),
                    layers.Flatten(),
                    layers.Dense(512, activation="relu"),
                    layers.Dense(self.num_actions, activation="linear"),
                ]
            )
        elif self.model_type == "mlp":
            return keras.Sequential(
                [
                    layers.Dense(100, activation="relu", input_shape=(self.FEATURE_LENGTH,)),
                    layers.Dense(200, activation="relu"),
                    layers.Dense(self.num_actions, activation="linear"),
                ]
            )
        
        else:
            raise ValueError('Model type should be "mlp" or "cnn".')
    
    def update(self,):
        pass
    
    def _model_input(self, state):
        if self.model_type == "mlp":
            state = np.concatenate((state["waypoints"][:, 0],
                                    np.array([state["d"]]),
                                    np.array([state["phi"]]),
                                    np.array([state["v_kmh"]])))
        if self.model_type == "cnn":
            state = state["image"]
        
        return state
    
    def train(self,):
        model = self.create_q_model()
        model_target = self.create_q_model()
        # Experience history
        action_history = []
        state_history = []
        state_next_history = []
        reward_history = []
        done_history = []
        episode_reward_history = []
        running_reward = 0
        episode_count = 0
        frame_count = 0
        
        # Number of frames to take random action and observe output
        epsilon_random_frames = 500
        # Number of frames for exploration
        epsilon_greedy_frames = 1000.0
        # Maximum replay length
        # Note: The Deepmind paper suggests 1000000 however this causes memory issues
        max_memory_length = 100000
        # Train the model after 4 actions
        update_after_actions = 4
        # How often to update the target network
        update_target_network = 1000
        
        # initialize the environment
        env = CarlaEnv()
        
        while True:
            # resetting the environment
            state = env.reset()
            state = self._model_input(state)
            
            episode_reward = 0
            
            for timestep in range(1, self.max_steps_per_episode):
                frame_count += 1
                
                # taking epsilon-greedy action
                if frame_count < epsilon_random_frames or self.epsilon > np.random.rand(1)[0]:
                    action = np.random.choice(self.num_actions)
                else:
                # taking actions based on Q-value estimations
                    action_probs = model(state, training=False)
                    # take the best action
                    action = tf.argmax(action_probs)
                    
                # decay the epsilon
                self.epsilon -= self.epsilon_interval/epsilon_greedy_frames
                self.epsilon = max(self.epsilon, self.min_epsilon)
                
                # taking the action
                next_state, reward, done, _ = env.step(action)
                next_state = self._model_input(next_state)
                episode_reward += reward
                
                # saving the actions, rewards and states
                action_history.append(action)
                state_history.append(state)
                state_next_history.append(next_state)
                done_history.append(done)
                reward_history.append(reward)
                
                # update the state
                state = next_state

                # Update every fourth frame and once batch size is over 32
                if frame_count % update_after_actions == 0 and len(done_history) > self.batch_size:
                    # Get indices of samples for replay buffers
                    indices = np.random.choice(range(len(done_history)), size=self.batch_size)

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
                    updated_q_values = rewards_sample + self.gamma * tf.reduce_max(future_rewards, axis=1)

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
                if frame_count % update_target_network == 0:
                    # update the the target network with new weights
                    model_target.set_weights(model.get_weights())
                    # Log details
                    template = "running reward: {:.2f} at episode {}, frame count {}"
                    print(template.format(running_reward, episode_count, frame_count))

                # Limit the state and reward history
                if len(reward_history) > max_memory_length:
                    del reward_history[:1]
                    del state_history[:1]
                    del state_next_history[:1]
                    del action_history[:1]
                    del done_history[:1]

                if done:
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
                self.max_episodes > 0 and episode_count >= self.max_episodes
            ):  # Maximum number of episodes reached
                print("Stopped at episode {}!".format(episode_count))
                break
    
    def predict(self,):
        pass