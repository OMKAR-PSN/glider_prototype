import random
import math
from typing import Tuple

class WindModel:
    """
    Randomizable wind for Monte Carlo testing.
    """
    def __init__(self, speed_m_s: float = 0.0, direction_rad: float = 0.0):
        self.speed = speed_m_s
        # CONVENTION: direction_rad uses the mathematical "blowing toward" convention.
        # e.g., direction_rad=0 means wind is blowing towards the positive X axis (East).
        # This matches sim/dynamics.py, training/env.py, and sim/scenario_validator.py.
        self.direction = direction_rad

    def randomize(self, max_speed: float = 5.0):
        self.speed = random.uniform(0.0, max_speed)
        self.direction = random.uniform(0.0, 2 * math.pi)

    def get_wind(self) -> Tuple[float, float]:
        """Returns wind components (wx, wy)"""
        wx = self.speed * math.cos(self.direction)
        wy = self.speed * math.sin(self.direction)
        return wx, wy
