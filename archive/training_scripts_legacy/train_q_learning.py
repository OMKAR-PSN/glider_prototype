import numpy as np
import pickle
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.env import GliderEnv

def discretize_state(obs):
    # Very coarse discretization for tabular Q-learning
    heading_err, dist, alt_excess, wind_speed, wind_dir, roll = obs
    
    h_bin = round(math.degrees(heading_err) / 45.0) * 45.0
    d_bin = round(dist / 200.0) * 200.0
    a_bin = round(alt_excess / 50.0) * 50.0
    
    return (h_bin, d_bin, a_bin)

def train_q_learning():
    env = GliderEnv()
    q_table = {}
    
    alpha = 0.1
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    min_epsilon = 0.1
    
    episodes = 500
    
    actions = [-math.radians(10), 0.0, math.radians(10)] # -10, 0, 10 deg/s
    
    for ep in range(episodes):
        obs = env.reset()
        state = discretize_state(obs)
        
        if state not in q_table:
            q_table[state] = np.zeros(len(actions))
            
        done = False
        total_reward = 0
        
        while not done:
            if np.random.random() < epsilon:
                action_idx = np.random.randint(len(actions))
            else:
                action_idx = np.argmax(q_table[state])
                
            action = actions[action_idx]
            
            next_obs, reward, done, _ = env.step(action)
            next_state = discretize_state(next_obs)
            
            if next_state not in q_table:
                q_table[next_state] = np.zeros(len(actions))
                
            best_next_q = np.max(q_table[next_state])
            q_table[state][action_idx] += alpha * (reward + gamma * best_next_q - q_table[state][action_idx])
            
            state = next_state
            total_reward += reward
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        if (ep + 1) % 50 == 0:
            print(f"Episode {ep+1}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    # Save model
    with open("models/q_table.npy", "wb") as f:
        pickle.dump(q_table, f)
    print("Saved Q-table to models/q_table.npy")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    train_q_learning()
