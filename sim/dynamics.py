import math
from enum import Enum, auto

class SimPhase(Enum):
    ROCKET = auto()
    DROGUE = auto()
    GLIDER = auto()

class GliderDynamics:
    """
    Physics simulation for Rocket Boost, Drogue Descent, and Glider Flight.
    """
    def __init__(self, initial_x: float, initial_y: float, initial_alt: float, initial_heading: float):
        self.x = initial_x
        self.y = initial_y
        self.altitude = initial_alt
        self.heading = initial_heading
        
        self.vertical_velocity = 0.0
        self.phase = SimPhase.ROCKET if initial_alt < 1.0 else SimPhase.GLIDER
        self.time_since_launch = 0.0
        
        self.airspeed = 15.0  # m/s
        self.glide_ratio = 5.0
        
        self.roll = 0.0
        self.pitch = 0.0
        
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.yaw_rate = 0.0
        
        self.gravity = 9.81
        self.drogue_deployed = False

        # Servo state
        self.left_pwm = 90.0
        self.right_pwm = 90.0

    def set_servos(self, left: float, right: float):
        self.left_pwm = left
        self.right_pwm = right

    def step(self, dt: float, wind_x: float, wind_y: float):
        """
        Steps the simulation forward by dt.
        """
        if self.altitude <= 0 and self.time_since_launch > 5.0:
            self.altitude = 0.0
            return # Landed
            
        self.time_since_launch += dt
        
        if self.phase == SimPhase.ROCKET:
            # 3 second motor burn
            thrust = 50.0 if self.time_since_launch < 3.0 else 0.0
            self.vertical_velocity += (thrust - self.gravity) * dt
            self.altitude += self.vertical_velocity * dt
            
            # Wind drift
            self.x += wind_x * dt
            self.y += wind_y * dt
            
            # Apogee detection
            if thrust == 0.0 and self.vertical_velocity < 0:
                self.phase = SimPhase.DROGUE
                self.drogue_deployed = True
                
        elif self.phase == SimPhase.DROGUE:
            # Pull velocity to terminal velocity of -10 m/s
            self.vertical_velocity += (-10.0 - self.vertical_velocity) * dt * 0.5
            self.altitude += self.vertical_velocity * dt
            
            # Wind drift (drogues drift heavily)
            self.x += wind_x * dt
            self.y += wind_y * dt
            
            # Glider deployment at 600m
            if self.altitude <= 600.0:
                self.phase = SimPhase.GLIDER
                
        elif self.phase == SimPhase.GLIDER:
            # Parafoil actuator dynamics (PWM -> Brakes)
            # Both servos neutral at 90. Range is 60 to 120.
            # Asymmetric brake (delta_a) drives turn rate.
            # Symmetric brake (delta_s) drives glide ratio/speed.
            delta_a_deg = (self.right_pwm - self.left_pwm) / 2.0
            delta_s_deg = (self.right_pwm + self.left_pwm) / 2.0 - 90.0
            
            # K_turn relates asymmetric brake (deg) to yaw rate (rad/s)
            # Assume max 30 deg brake gives ~30 deg/sec turn rate (0.5 rad/s)
            K_turn = 0.5 / 30.0 
            cmd_yaw_rate = K_turn * delta_a_deg
            
            # Simple first-order response for yaw rate
            self.yaw_rate += (cmd_yaw_rate - self.yaw_rate) * dt * 5.0
            
            self.heading += self.yaw_rate * dt
            self.heading = (self.heading + math.pi) % (2*math.pi) - math.pi
            
            # Roll and pitch are damped by the pendulum effect of the payload
            self.roll = 0.0
            self.pitch = 0.0
            self.roll_rate = 0.0
            self.pitch_rate = 0.0
            
            # Baseline performance from CAN-7U-SAT PDR
            base_glide = 4.0
            base_airspeed = 12.0
            
            # Symmetric brake slows the parafoil and decreases glide ratio
            # e.g., max symmetric brake (30 deg) drops airspeed by 2 m/s and glide ratio by 1.0
            self.airspeed = base_airspeed - (max(0.0, delta_s_deg) / 30.0) * 2.0
            self.glide_ratio = base_glide - (max(0.0, delta_s_deg) / 30.0) * 1.0
            
            # Kinematics
            self.ground_speed_x = self.airspeed * math.cos(self.heading) + wind_x
            self.ground_speed_y = self.airspeed * math.sin(self.heading) + wind_y
            
            self.x += self.ground_speed_x * dt
            self.y += self.ground_speed_y * dt
            
            # Altitude
            sink_rate = self.airspeed / self.glide_ratio
            self.vertical_velocity = -sink_rate
            self.altitude += self.vertical_velocity * dt
            
            if self.altitude < 0:
                self.altitude = 0.0
