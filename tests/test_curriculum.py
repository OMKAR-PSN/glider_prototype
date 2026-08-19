import sys
import os
# pyrefly: ignore [missing-import]
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from training.train_sac import CurriculumCallback

class DummyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.difficulty_stage = 1
        self.step_count = 0

    def reset(self, seed=None, options=None):
        return np.zeros(12, dtype=np.float32), {}

    def step(self, action):
        # Always return success (miss_distance = 10.0)
        # Episodes end every step to quickly accumulate 50 episodes.
        self.step_count += 1
        obs = np.zeros(12, dtype=np.float32)
        reward = 1.0
        terminated = True
        truncated = False
        info = {'miss_distance': 10.0}
        return obs, reward, terminated, truncated, info

def test_rolling_curriculum():
    env = DummyVecEnv([lambda: DummyEnv()])
    
    # We use a tiny SAC model so it runs fast
    model = SAC("MlpPolicy", env, learning_starts=0, train_freq=1, gradient_steps=1)
    
    callback = CurriculumCallback(total_timesteps=100)
    
    print("\n--- Starting Curriculum Sanity Run ---")
    # Run for 60 steps. 
    # At step 50, we should have 50 episodes of history (100% success rate).
    # It should trigger Stage 2 and flush the buffer.
    model.learn(total_timesteps=60, callback=callback)
    
    print(f"\nFinal stage after 60 steps: {callback.current_stage}")
    assert callback.current_stage == 2, f"Expected stage 2, got {callback.current_stage}"
    print("[PASS] Curriculum advanced to Stage 2 after 50 successful episodes.")
    
    print("\n--- Continuing Curriculum Sanity Run ---")
    # Now run for another 60 steps. Since history was cleared, it will take
    # another 50 steps (total 100) to trigger Stage 3.
    model.learn(total_timesteps=60, callback=callback, reset_num_timesteps=False)
    
    print(f"\nFinal stage after 120 steps: {callback.current_stage}")
    assert callback.current_stage == 3, f"Expected stage 3, got {callback.current_stage}"
    print("[PASS] Curriculum advanced to Stage 3 after another 50 successful episodes.")

if __name__ == "__main__":
    test_rolling_curriculum()
