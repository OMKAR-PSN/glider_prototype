import sys
import os
import math
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.run_sim import SimRunner
from sim.dynamics import SimPhase

from sim.scenario_validator import check_scenario_reachability

def run_monte_carlo(num_drops: int = 500, use_rl: bool = False, model_path: str = "models/sac_model.zip"):
    print(f"Running Monte Carlo Simulation: {num_drops} drops")
    print(f"Controller: {'Reinforcement Learning (SAC)' if use_rl else 'Traditional Math (PID)'}")
    print("Randomizing: Wind (0-8 m/s), Airspeed (10-18 m/s), Glide Ratio (3-6), Drop position (500-1500m)")
    
    # Set a fixed seed to guarantee identical drops between PID and SAC runs
    np.random.seed(42)
    
    miss_distances = []
    unreachable_headwind = 0
    unreachable_distance = 0
    
    model = None
    if use_rl:
        from stable_baselines3 import SAC
        try:
            # Explicitly force CPU inference to avoid GPU transfer overhead on tiny MLPs
            model = SAC.load(model_path, device='cpu')
        except FileNotFoundError:
            print(f"Error: {model_path} not found!")
            return float('inf')
            
    for _ in tqdm(range(num_drops)):
        # Generate scenario variables first
        r = np.random.uniform(500, 1500)
        theta = np.random.uniform(0, 2 * math.pi)
        initial_x = r * math.cos(theta)
        initial_y = r * math.sin(theta)
        
        airspeed = np.random.uniform(10.0, 18.0)
        glide_ratio = np.random.uniform(3.0, 6.0)
        
        # Wind is randomized 0-8 in WindModel, but let's pre-generate to validate
        wind_speed = np.random.uniform(0.0, 8.0)
        wind_dir = np.random.uniform(0, 2 * math.pi)
        
        distance_to_target = math.hypot(initial_x, initial_y)
        bearing_to_target = math.atan2(-initial_y, -initial_x)
        
        is_reachable, reason = check_scenario_reachability(
            altitude=600.0, 
            glide_ratio=glide_ratio, 
            airspeed=airspeed, 
            distance_to_target=distance_to_target, 
            wind_speed=wind_speed, 
            wind_direction=wind_dir, 
            bearing_to_target=bearing_to_target
        )
        
        if not is_reachable:
            if reason == "headwind_exceeds_airspeed":
                unreachable_headwind += 1
            elif reason == "too_far_for_glide_ratio":
                unreachable_distance += 1
            continue
            
        # If reachable, run the simulation
        if use_rl:
            from training.env import GliderEnv
            env = GliderEnv()
            # Override env randomizations to match our validated scenario
            obs, _ = env.reset()
            env.dynamics.x = initial_x
            env.dynamics.y = initial_y
            env.dynamics.airspeed = airspeed
            env.dynamics.glide_ratio = glide_ratio
            env.wind.speed = wind_speed
            env.wind.direction = wind_dir
            # Re-generate first obs with overridden values
            obs = env._get_obs()
            
            done = False
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
            # Compute miss distance directly from dynamics state (not from obs indices)
            import math as _math
            dist = _math.hypot(env.dynamics.x - env.target_x, env.dynamics.y - env.target_y)
            miss_distances.append(dist)
            
        else:
            runner = SimRunner()
            runner.dynamics.x = initial_x
            runner.dynamics.y = initial_y
            runner.dynamics.altitude = 600.0
            runner.dynamics.phase = SimPhase.GLIDER
            
            from state_machine.flight_states import FlightState
            runner.state_machine.state = FlightState.GUIDED_DESCENT
            
            runner.dynamics.heading = math.radians(np.random.uniform(0, 360))
            runner.dynamics.airspeed = airspeed
            runner.dynamics.glide_ratio = glide_ratio
            runner.wind.speed = wind_speed
            runner.wind.direction = wind_dir
            
            running = True
            while running:
                running = runner.step()
                
            final_x = runner.dynamics.x
            final_y = runner.dynamics.y
            dist = math.hypot(final_x - runner.target_x, final_y - runner.target_y)
            miss_distances.append(dist)
        
    miss_distances = np.array(miss_distances)
    reachable_drops = len(miss_distances)
    
    # Calculate CEP (Circular Error Probable)
    if reachable_drops > 0:
        cep50 = np.percentile(miss_distances, 50)
        cep90 = np.percentile(miss_distances, 90)
        max_miss = np.max(miss_distances)
        success_rate = np.mean(miss_distances <= 20.0) * 100.0
    else:
        cep50 = cep90 = max_miss = success_rate = 0.0
    
    print("\n--- Monte Carlo Results ---")
    print(f"Total Drops Attempted: {num_drops}")
    print(f"Unreachable (Headwind Exceeds Airspeed): {unreachable_headwind}")
    print(f"Unreachable (Too Far for Glide Ratio): {unreachable_distance}")
    print(f"Reachable Drops Evaluated: {reachable_drops}")
    print("---------------------------")
    print(f"Success Rate (< 20m miss): {success_rate:.1f}%")
    print(f"CEP50 (Median Miss): {cep50:.2f} m")
    print(f"CEP90 (90% landed within): {cep90:.2f} m")
    print(f"Max Miss Distance: {max_miss:.2f} m")
    
    if reachable_drops > 0 and cep90 <= 15.0:
        print("\n[PASSED] CEP90 is <= 15 meters on reachable subset.")
    else:
        print("\n[FAILED] CEP90 is > 15 meters on reachable subset. Tuning required.")
        
    return cep50

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Glider GNC Monte Carlo Simulation")
    parser.add_argument("--drops", type=int, default=500, help="Number of simulated drops")
    parser.add_argument("--rl", action="store_true", help="Test the Reinforcement Learning model instead of PID")
    parser.add_argument("--model", type=str, default="models/sac_model.zip", help="Path to RL model zip")
    args = parser.parse_args()
    
    run_monte_carlo(args.drops, args.rl, args.model)
