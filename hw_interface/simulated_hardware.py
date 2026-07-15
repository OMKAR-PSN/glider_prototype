import math
from hw_interface.base import HWInterface
from sensors.drivers import IMUData, BaroData, GPSData, PowerData

class SimulatedHardware(HWInterface):
    """
    Implements the hardware interface by reading from the simulation dynamics model.
    Used for pure simulation and HIL testing.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics

    def initialize(self):
        pass

    def read_imu(self) -> IMUData:
        # Simulate IMU readings based on dynamics state
        # In a real sim, add noise. Compute gravity vector in body frame.
        g = 9.81
        ax = -g * math.sin(self.dynamics.pitch)
        ay = g * math.cos(self.dynamics.pitch) * math.sin(self.dynamics.roll)
        az = -g * math.cos(self.dynamics.pitch) * math.cos(self.dynamics.roll) # Negative because Z is down? Wait, usually Z is down so gravity is negative when level?
        # Body frame: X forward, Y right, Z down.
        # When level, gravity points DOWN, so accel_z = +9.81
        phi = self.dynamics.roll
        theta = self.dynamics.pitch
        
        p = self.dynamics.roll_rate - self.dynamics.yaw_rate * math.sin(theta)
        q = self.dynamics.pitch_rate * math.cos(phi) + self.dynamics.yaw_rate * math.sin(phi) * math.cos(theta)
        r = -self.dynamics.pitch_rate * math.sin(phi) + self.dynamics.yaw_rate * math.cos(phi) * math.cos(theta)

        return IMUData(
            accel_x=-g * math.sin(theta),
            accel_y=g * math.sin(phi) * math.cos(theta),
            accel_z=g * math.cos(theta) * math.cos(phi),
            gyro_p=p,
            gyro_q=q,
            gyro_r=r,
            mag_x=math.cos(self.dynamics.heading),
            mag_y=math.sin(self.dynamics.heading),
            mag_z=0.0
        )

    def read_baro(self) -> BaroData:
        # Use simple standard atmosphere conversion
        altitude = self.dynamics.altitude
        pressure = 101325.0 * (1 - 2.25577e-5 * altitude)**5.25588
        return BaroData(pressure=pressure, temperature=20.0, altitude=altitude)

    def read_gps(self) -> GPSData:
        # Calculate true ground velocity from dynamics state
        # In a real sim we'd track previous positions, but we can compute it since dynamics exposes heading
        # Wait, SimulatedHardware doesn't have wind. Let's just track position derivatives.
        # Alternatively, we can use the exact dx/dy if dynamics stores them.
        v_gx = getattr(self.dynamics, 'ground_speed_x', self.dynamics.airspeed * math.cos(self.dynamics.heading))
        v_gy = getattr(self.dynamics, 'ground_speed_y', self.dynamics.airspeed * math.sin(self.dynamics.heading))
        
        return GPSData(
            latitude=self.dynamics.x * 1e-5,  # fake conversion
            longitude=self.dynamics.y * 1e-5, # fake conversion
            altitude=self.dynamics.altitude,
            ground_speed=math.hypot(v_gx, v_gy),
            heading=math.atan2(v_gy, v_gx),
            fix=True
        )

    def read_power(self) -> PowerData:
        return PowerData(5.0, 0.5)

    def write_servos(self, left_pwm: float, right_pwm: float):
        # Pass the servo commands to the dynamics model
        self.dynamics.set_servos(left_pwm, right_pwm)

    def trigger_drogue(self):
        self.dynamics.drogue_deployed = True
