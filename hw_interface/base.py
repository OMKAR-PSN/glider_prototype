from abc import ABC, abstractmethod
from typing import Tuple, Optional
from sensors.drivers import IMUData, BaroData, GPSData, PowerData

class HWInterface(ABC):
    """
    Abstract interface for hardware.
    Provides methods to read all sensors and write to servos.
    """
    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def read_imu(self) -> Optional[IMUData]:
        pass

    @abstractmethod
    def read_baro(self) -> Optional[BaroData]:
        pass

    @abstractmethod
    def read_gps(self) -> Optional[GPSData]:
        pass
        
    @abstractmethod
    def read_power(self) -> Optional[PowerData]:
        pass

    @abstractmethod
    def write_servos(self, left_pwm: float, right_pwm: float):
        """
        Write PWM values to the elevon servos.
        Args:
            left_pwm: Angle in degrees
            right_pwm: Angle in degrees
        """
        pass
        
    @abstractmethod
    def trigger_drogue(self):
        """Deploy drogue chute."""
        pass
