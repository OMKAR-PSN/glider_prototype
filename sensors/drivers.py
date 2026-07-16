from dataclasses import dataclass
from typing import Optional

from hw_interface.pinmap import BMP388_SPI, BNO085_SPI, INA219_I2C, LORA_UART

@dataclass
class IMUData:
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_p: float
    gyro_q: float
    gyro_r: float
    mag_x: float
    mag_y: float
    mag_z: float

class BNO085:
    """Stub driver for the BNO085 IMU on SPI0."""
    def __init__(self, bus: int = BNO085_SPI.bus, device: int = BNO085_SPI.device,
                 cs_gpio: int = BNO085_SPI.cs_gpio,
                 int_gpio: int | None = BNO085_SPI.int_gpio,
                 rst_gpio: int | None = BNO085_SPI.rst_gpio):
        self.bus = bus
        self.device = device
        self.cs_gpio = cs_gpio
        self.int_gpio = int_gpio
        self.rst_gpio = rst_gpio

    def read(self) -> Optional[IMUData]:
        # Implement BNO085 SPI reading logic here.
        return IMUData(0.0, 0.0, 9.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

ICM20948 = BNO085

@dataclass
class BaroData:
    pressure: float
    temperature: float
    altitude: float

class BMP388:
    """Stub driver for the BMP388 barometer on SPI0."""
    def __init__(self, bus: int = BMP388_SPI.bus, device: int = BMP388_SPI.device,
                 cs_gpio: int = BMP388_SPI.cs_gpio,
                 int_gpio: int | None = BMP388_SPI.int_gpio):
        self.bus = bus
        self.device = device
        self.cs_gpio = cs_gpio
        self.int_gpio = int_gpio

    def read(self) -> Optional[BaroData]:
        # Implement BMP388 SPI reading logic here.
        return BaroData(101325.0, 20.0, 0.0)

@dataclass
class GPSData:
    latitude: float
    longitude: float
    altitude: float
    ground_speed: float
    heading: float
    fix: bool

class GPS:
    """Stub driver for a GNSS UART, if fitted separately from the HAT."""
    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate

    def read(self) -> Optional[GPSData]:
        # Implement UART reading and NMEA parsing logic here.
        return GPSData(0.0, 0.0, 0.0, 0.0, 0.0, False)

@dataclass
class PowerData:
    voltage: float
    current: float

class INA219:
    """Stub driver for the INA219 current sensor on I2C1."""
    def __init__(self, bus: int = INA219_I2C.bus, address: int = INA219_I2C.address):
        self.bus = bus
        self.address = address

    def read(self) -> Optional[PowerData]:
        return PowerData(5.0, 0.5)

class LoRaTelemetry:
    """UART LoRa module pins from connector J3."""
    def __init__(self, port: str = LORA_UART.port, baudrate: int = LORA_UART.baudrate,
                 m0_gpio: int | None = LORA_UART.m0_gpio,
                 m1_gpio: int | None = LORA_UART.m1_gpio,
                 aux_gpio: int | None = LORA_UART.aux_gpio):
        self.port = port
        self.baudrate = baudrate
        self.m0_gpio = m0_gpio
        self.m1_gpio = m1_gpio
        self.aux_gpio = aux_gpio
