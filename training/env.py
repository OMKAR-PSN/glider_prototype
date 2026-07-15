import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces
from sim.dynamics import GliderDynamics, SimPhase
from sim.wind_model import WindModel

class GliderEnv(gym.Env):
    """
    A stable-baselines3 compatible Gymnasium environment for training Parafoil RL guidance policies.
    """
    def __init__(self):
        super(GliderEnv, self).__init__()
        
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_alt = 0.0
        self.dt = 0.05
        self.max_steps = 4000
        
        self.difficulty_stage = 3 # Default to hardest stage
        
        # Actions: [delta_a (asymmetric brake -30 to 30), delta_s (symmetric brake 0 to 30)]
        self.action_space = spaces.Box(low=np.array([-30.0, 0.0]), high=np.array([30.0, 30.0]), dtype=np.float32)
        
        # Obs (16 dims): [sin(heading_err), cos(heading_err), dist, alt_excess, wind_speed,
        #                  sin(wind_dir), cos(wind_dir), pitch, roll, yaw_rate,
        #                  prev_delta_a, prev_delta_s,
        #                  sin(track_err), cos(track_err), lateral_drift, time_to_impact]
        #
        # track_err    = bearing(target) - course_over_ground (sin/cos to avoid angle wrap)
        # lateral_drift = ground_velocity component perpendicular to target bearing / 8.0
        # time_to_impact = altitude / sink_rate / 200.0
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if self.difficulty_stage == 1:
            wind_max = 2.0
            r_min, r_max = 200, 500
            airspeed_min, airspeed_max = 12.0, 12.0
            glide_min, glide_max = 4.5, 4.5
        elif self.difficulty_stage == 2:
            wind_max = 4.0
            r_min, r_max = 300, 800
            airspeed_min, airspeed_max = 11.0, 14.0
            glide_min, glide_max = 4.0, 5.0
        else: # Stage 3
            wind_max = 8.0
            r_min, r_max = 500, 1500
            airspeed_min, airspeed_max = 10.0, 18.0
            glide_min, glide_max = 3.0, 6.0
            
        # Randomize initial position
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * math.pi)
        initial_x = r * math.cos(theta)
        initial_y = r * math.sin(theta)
        
        # Start at 600m to skip rocket/drogue phase for pure glide training
        self.dynamics = GliderDynamics(initial_x, initial_y, 600.0, math.radians(np.random.uniform(0, 360)))
        self.dynamics.phase = SimPhase.GLIDER
        
        # Domain Randomization
        self.dynamics.airspeed = np.random.uniform(airspeed_min, airspeed_max)
        self.dynamics.glide_ratio = np.random.uniform(glide_min, glide_max)
        
        self.wind = WindModel()
        self.wind.randomize(max_speed=wind_max)
        
        self.step_count = 0
        
        # Initialize previous state variables
        self.previous_action = np.array([0.0, 0.0], dtype=np.float32) # Neutral trim
        self.previous_distance = math.hypot(self.target_x - initial_x, self.target_y - initial_y)
        
        return self._get_obs(), {}

    def _get_obs(self):
        dx = self.target_x - self.dynamics.x
        dy = self.target_y - self.dynamics.y
        dist = math.hypot(dx, dy)
        
        target_bearing = math.atan2(dy, dx)
        heading_err = (target_bearing - self.dynamics.heading + math.pi) % (2*math.pi) - math.pi
        
        alt_needed = dist / self.dynamics.glide_ratio
        alt_excess = self.dynamics.altitude - alt_needed
        
        # --- New obs 13-16: wind-aware navigation signals ---
        
        # Ground velocity = airspeed vector + wind vector
        wx, wy = self.wind.get_wind()
        gvx = self.dynamics.airspeed * math.cos(self.dynamics.heading) + wx
        gvy = self.dynamics.airspeed * math.sin(self.dynamics.heading) + wy
        
        # Track error: gap between course-over-ground and required bearing
        # Positive = COG is to the left of the target bearing (need to turn right)
        # Encoded as sin/cos to avoid angle-wrap discontinuity
        course_over_ground = math.atan2(gvy, gvx)
        track_err = (target_bearing - course_over_ground + math.pi) % (2 * math.pi) - math.pi
        
        # Lateral drift: ground velocity component perpendicular to target bearing
        # Positive = drift to the left of target line
        # Normalized by max stage wind speed (8 m/s) -> approx [-1, 1]
        lateral_drift = (-gvx * math.sin(target_bearing) + gvy * math.cos(target_bearing))
        lateral_drift_norm = lateral_drift / 8.0
        
        # Time to impact: urgency signal. sink_rate = airspeed / glide_ratio
        # Normalized by 200s (max flight time at 20Hz x max_steps=4000 steps)
        sink_rate = max(self.dynamics.airspeed / self.dynamics.glide_ratio, 0.1)  # guard div/0
        time_to_impact = self.dynamics.altitude / sink_rate
        time_to_impact_norm = min(time_to_impact / 200.0, 2.0)  # cap at 2.0 for edge cases
        
        # 16-dimensional observation
        return np.array([
            math.sin(heading_err),            # obs[0]  heading error (sin)
            math.cos(heading_err),            # obs[1]  heading error (cos)
            dist / 1000.0,                    # obs[2]  distance to target
            alt_excess / 1000.0,              # obs[3]  altitude margin
            self.wind.speed / 10.0,           # obs[4]  wind magnitude
            math.sin(self.wind.direction),    # obs[5]  wind direction (sin)
            math.cos(self.wind.direction),    # obs[6]  wind direction (cos)
            self.dynamics.pitch / 0.5,        # obs[7]  pitch rate
            self.dynamics.roll / 0.5,         # obs[8]  roll rate
            self.dynamics.yaw_rate / 0.5,     # obs[9]  yaw rate
            self.previous_action[0] / 30.0,   # obs[10] prev delta_a
            self.previous_action[1] / 30.0,   # obs[11] prev delta_s
            math.sin(track_err),              # obs[12] track error (sin) — COG vs bearing
            math.cos(track_err),              # obs[13] track error (cos) — COG vs bearing
            lateral_drift_norm,               # obs[14] crosswind drift rate
            time_to_impact_norm,              # obs[15] urgency / time remaining
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        delta_a = action[0]
        delta_s = action[1]
        
        # Mix to servo pwm limits [60, 120]
        left_pwm = 90.0 + delta_s - delta_a
        right_pwm = 90.0 + delta_s + delta_a
        left_pwm = max(60.0, min(120.0, left_pwm))
        right_pwm = max(60.0, min(120.0, right_pwm))
        
        self.dynamics.set_servos(left_pwm, right_pwm)
        
        wx, wy = self.wind.get_wind()
        self.dynamics.step(self.dt, wx, wy)
        
        dist_meters = math.hypot(self.target_x - self.dynamics.x, self.target_y - self.dynamics.y)
        
        # Fix 2: Potential-based reward shaping
        reward = (self.previous_distance - dist_meters)
        self.previous_distance = dist_meters
        
        # Fix 4: Action smoothness penalty (scaled down for 20Hz dt)
        action_delta_penalty = 0.025 * np.sum(np.abs(action - self.previous_action))
        reward -= action_delta_penalty
        
        self.previous_action = action.copy()
        
        obs = self._get_obs()
        
        terminated = False
        truncated = False
        
        if self.dynamics.altitude <= 0:
            terminated = True
            # Massive Sparse terminal bonuses
            if dist_meters < 10.0:
                reward += 10000.0
            elif dist_meters < 20.0:
                reward += 2000.0
            elif dist_meters < 50.0:
                reward += 200.0
                
        if self.step_count >= self.max_steps:
            truncated = True
                
        info = {}
        if terminated or truncated:
            info['miss_distance'] = dist_meters
            
        return obs, reward, terminated, truncated, info
