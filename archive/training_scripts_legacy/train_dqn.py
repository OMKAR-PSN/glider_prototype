import os
import math
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.env import GliderEnv
import numpy as np

try:
    import gym
    from gym import spaces
    from stable_baselines3 import DQN
except ImportError:
    print("stable-baselines3 or gym not installed. Cannot run DQN training.")
    exit(0)

class GymDiscreteWrapper(gym.Env):
    def __init__(self):
        super(GymDiscreteWrapper, self).__init__()
        self.env = GliderEnv()
        
        # Action space: 0: -10 deg/s, 1: 0, 2: +10 deg/s
        self.action_space = spaces.Discrete(3)
        
        # Obs: [heading_error, distance_to_target, altitude_excess, wind_speed, wind_direction, current_bank_angle]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        action_map = {0: -10.0, 1: 0.0, 2: 10.0}
        turn_rate = math.radians(action_map.get(int(action), 0.0))
        obs, reward, done, info = self.env.step(turn_rate)
        return obs, reward, done, info

def train_dqn():
    os.makedirs("models", exist_ok=True)
    env = GymDiscreteWrapper()
    
    model = DQN("MlpPolicy", env, verbose=1)
    print("Training DQN for 10000 steps...")
    model.learn(total_timesteps=10000)
    
    model.save("models/dqn_model.zip")
    print("Saved DQN model to models/dqn_model.zip")

if __name__ == "__main__":
    train_dqn()
