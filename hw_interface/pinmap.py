"""
GARUD HAT pin mapping from Schema_Draft_2.pdf.

Numbering uses Broadcom GPIO numbers, not physical header pin numbers.
The HAT routes BMP388 over SPI0, BNO085, PCA9685 and INA219 over I2C1,
LoRa telemetry over UART0, and a transistor-driven buzzer on GPIO16.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpiDevicePins:
    bus: int
    device: int
    cs_gpio: int
    int_gpio: int | None = None
    rst_gpio: int | None = None


@dataclass(frozen=True)
class I2cDevicePins:
    bus: int
    address: int
    sda_gpio: int = 2
    scl_gpio: int = 3


@dataclass(frozen=True)
class UartPins:
    port: str
    baudrate: int
    tx_gpio: int
    rx_gpio: int
    m0_gpio: int | None = None
    m1_gpio: int | None = None
    aux_gpio: int | None = None


# Shared Raspberry Pi interfaces.
I2C1_BUS = 1
SPI0_BUS = 0
SPI0_MOSI_GPIO = 10
SPI0_MISO_GPIO = 9
SPI0_SCLK_GPIO = 11

# Sensors.
BNO085_I2C = I2cDevicePins(bus=I2C1_BUS, address=0x4A)
BMP388_SPI = SpiDevicePins(bus=SPI0_BUS, device=0, cs_gpio=8)

# I2C devices. Verify one of these addresses is changed on hardware if both
# modules are populated, because many PCA9685 and INA219 breakouts default to 0x40.
PCA9685_I2C = I2cDevicePins(bus=I2C1_BUS, address=0x40)
INA219_I2C = I2cDevicePins(bus=I2C1_BUS, address=0x41)

# Servo controller connector J2.
SERVO_OE_GPIO = 4
LEFT_SERVO_CHANNEL = 0
RIGHT_SERVO_CHANNEL = 1
DROGUE_SERVO_CHANNEL = 2

# Telemetry connector J3.
LORA_UART = UartPins(
    port="/dev/serial0",
    baudrate=9600,
    tx_gpio=14,
    rx_gpio=15,
    m0_gpio=23,
    m1_gpio=24,
    aux_gpio=25,
)

# Buzzer circuit.
BUZZER_GPIO = 16
