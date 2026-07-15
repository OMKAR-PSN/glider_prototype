import numpy as np

class RLGuidance:
    """
    Optional RL-based guidance (Q-learning, DQN, SAC).
    Outputs a commanded turn rate, which will be converted to bank angle.
    """
    _warning_printed = False

    def __init__(self, model_path: str, policy_type: str):
        self.model_path = model_path
        self.policy_type = policy_type
        self.model = None
        self._load_model()

    def _load_model(self):
        """Loads the corresponding policy."""
        if self.policy_type == "tabular":
            try:
                self.model = np.load(self.model_path, allow_pickle=True).item()
            except FileNotFoundError:
                print(f"Warning: Model not found at {self.model_path}")
                self.model = {}
        elif self.policy_type in ["dqn", "sac"]:
            try:
                from stable_baselines3 import DQN, SAC
                if self.policy_type == "dqn":
                    self.model = DQN.load(self.model_path)
                else:
                    self.model = SAC.load(self.model_path)
            except ImportError:
                if not RLGuidance._warning_printed:
                    print("stable-baselines3 not installed. RL models cannot be loaded.")
                    RLGuidance._warning_printed = True
            except Exception as e:
                if not RLGuidance._warning_printed:
                    print(f"Failed to load RL model: {e}")
                    RLGuidance._warning_printed = True

    def compute(self, obs: np.ndarray) -> float:
        """
        Computes the action given an observation.
        Observation: [heading_error, distance_to_target, altitude_excess, wind_speed, wind_direction, current_bank_angle]
        
        Returns:
            Commanded turn rate in rad/s
        """
        if self.model is None:
            return 0.0

        if self.policy_type == "tabular":
            # Discretize observation to match training (simplified)
            state_key = tuple(np.round(obs, decimals=1))
            if state_key in self.model:
                action = np.argmax(self.model[state_key])
            else:
                action = 1 # Default neutral action if unvisited state
            
            # Action space: 0: -10 deg/s, 1: 0 deg/s, 2: +10 deg/s
            action_map = {0: -10.0, 1: 0.0, 2: 10.0}
            turn_rate_deg_s = action_map.get(action, 0.0)
            return np.deg2rad(turn_rate_deg_s)
            
        elif self.policy_type in ["dqn", "sac"]:
            # SB3 models
            action, _states = self.model.predict(obs, deterministic=True)
            if self.policy_type == "dqn":
                action_map = {0: -10.0, 1: 0.0, 2: 10.0}
                turn_rate_deg_s = action_map.get(int(action), 0.0)
                return np.deg2rad(turn_rate_deg_s)
            else:
                # SAC is continuous [-1, 1], scale to [-15, 15] deg/s (example scale)
                turn_rate_deg_s = float(action[0]) * 15.0
                return np.deg2rad(turn_rate_deg_s)

        return 0.0
