import numpy as np
import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import gymnasium as gym
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
from collections import deque
import random



# Environment setup
env = gym.make("CartPole-v1", render_mode="human")
num_states = env.observation_space.shape[0]
num_actions = env.action_space.n

# Hyperparameters
gamma = 0.99  # Discount factor for future rewards
epsilon = 1.0  # Exploration rate in the epsilon-greedy policy
epsilon_min = 0.01  # Minimum value to which epsilon can decay
epsilon_decay = 0.995  # Decay rate for exploration after each episode 
learning_rate = 0.001  # (alpha) Step size for the optimizer
batch_size = 64  # Batch size for experience replay to train network in one iteration
max_steps_per_episode = 200 # Max Steps agent can take before it terminates the episode
total_episodes = 100 # Total number of episodes agent will be trained
target_update_freq = 10  # Frequency of updating the target network

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        return len(self.buffer)

# Create the Q-network
def create_q_model():
    model = models.Sequential()
    model.add(layers.Dense(64, input_dim=num_states, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_actions, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return model

# Initialize Q-networks
q_model = create_q_model()
target_q_model = create_q_model()
target_q_model.set_weights(q_model.get_weights())


# Experience replay buffer
replay_buffer = ReplayBuffer()

# Function to update the target Q-network
def update_target_network():
    target_q_model.set_weights(q_model.get_weights())

# Function to choose an action using epsilon-greedy policy
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.choice(num_actions)
    q_values = q_model.predict(state)
    return np.argmax(q_values[0])

# Policy for action selection
def policy(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(num_actions)
    q_values = q_model.predict(state, verbose=0)
    return np.argmax(q_values[0])

# Load previously saved weights if available
try:
    q_model.load_weights("cartpole_dqn.weights.h5")
    target_q_model.load_weights("cartpole_target_dqn.weights.h5")
    
    print("Loaded weights from file")

except:
    print("No weights file found, starting training from scratch")

# Training the DQN
reward_list = []
for episode in range(total_episodes):
    state, _ = env.reset()
    state = np.reshape(state, [1, num_states])
    total_reward = 0

    for step in range(max_steps_per_episode):
        action = choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = np.reshape(next_state, [1, num_states])
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        
        if done or truncated:
            print(f"Episode: {episode+1}/{total_episodes}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")
            break
        
        if replay_buffer.size() > batch_size:
            minibatch = replay_buffer.sample(batch_size)
            states = np.array([experience[0] for experience in minibatch])
            actions = np.array([experience[1] for experience in minibatch])
            rewards = np.array([experience[2] for experience in minibatch])
            next_states = np.array([experience[3] for experience in minibatch])
            dones = np.array([experience[4] for experience in minibatch])

            states = np.squeeze(states)
            next_states = np.squeeze(next_states)

            q_values_next = target_q_model.predict(next_states)
            targets = rewards + (1 - dones) * gamma * np.amax(q_values_next, axis=1)
            targets_full = q_model.predict(states)
            
            for i, action in enumerate(actions):
                targets_full[i][action] = targets[i]

            q_model.fit(states, targets_full, epochs=1, verbose=0)

    reward_list.append(total_reward)
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    if (episode + 1) % 10 == 0:
        update_target_network()

# Save the weights
q_model.save_weights("cartpole_dqn.weights.h5")
target_q_model.save_weights("cartpole_target_dqn.weights.h5")

# Plotting the rewards
plt.plot(range(total_episodes),reward_list)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()
