from hw_interface.base import HWInterface
from hw_interface.pinmap import (
    BUZZER_GPIO,
    DROGUE_SERVO_CHANNEL,
    LEFT_SERVO_CHANNEL,
    PCA9685_I2C,
    RIGHT_SERVO_CHANNEL,
    SERVO_OE_GPIO,
)
from sensors.drivers import IMUData, BaroData, GPSData, PowerData, BNO085, BMP388, GPS, INA219

class RealHardware(HWInterface):
    """
    Implements the GARUD HAT hardware interface for the physical Raspberry Pi 4.

    Pin mapping follows Schema_Draft_2.pdf:
    - BNO085: SPI0, CS=GPIO5, RST=GPIO6, INT=GPIO13
    - BMP388: SPI0, CS=GPIO22, INT=GPIO27
    - PCA9685 servo controller: I2C1, OE=GPIO4
    - INA219: I2C1
    - Buzzer: GPIO16
    """
    def __init__(self):
        self.imu = BNO085()
        self.baro = BMP388()
        self.gps = GPS()
        self.power = INA219()
        self.servo_oe_gpio = SERVO_OE_GPIO
        self.buzzer_gpio = BUZZER_GPIO
        self.left_servo_channel = LEFT_SERVO_CHANNEL
        self.right_servo_channel = RIGHT_SERVO_CHANNEL
        self.drogue_servo_channel = DROGUE_SERVO_CHANNEL
        self._servo_kit = None
        self._gpio = None
        self.last_servo_write = (90.0, 90.0)

    def initialize(self):
        print("Initializing GARUD HAT sensors and PWM...")
        self._init_gpio()
        self._init_servos()

    def _init_gpio(self):
        try:
            import RPi.GPIO as GPIO
        except ImportError:
            self._gpio = None
            return

        self._gpio = GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.servo_oe_gpio, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.buzzer_gpio, GPIO.OUT, initial=GPIO.LOW)

    def _init_servos(self):
        try:
            from adafruit_servokit import ServoKit
        except ImportError:
            self._servo_kit = None
            return

        self._servo_kit = ServoKit(channels=16, address=PCA9685_I2C.address)
        if self._gpio:
            self._gpio.output(self.servo_oe_gpio, self._gpio.LOW)

    def read_imu(self) -> IMUData:
        return self.imu.read()

    def read_baro(self) -> BaroData:
        return self.baro.read()

    def read_gps(self) -> GPSData:
        return self.gps.read()

    def read_power(self) -> PowerData:
        return self.power.read()

    def write_servos(self, left_pwm: float, right_pwm: float):
        """
        Write degrees to the PCA9685 servo controller.

        The schematic routes servo control through J2/PCA9685 over I2C1,
        with output-enable controlled by GPIO4. Servo angles are already
        bounded by the flight computer before this call.
        """
        self.last_servo_write = (left_pwm, right_pwm)
        if self._servo_kit is None:
            return

        self._servo_kit.servo[self.left_servo_channel].angle = left_pwm
        self._servo_kit.servo[self.right_servo_channel].angle = right_pwm
        
    def trigger_drogue(self):
        """Trigger real drogue release servo on the PCA9685."""
        if self._servo_kit is not None:
            self._servo_kit.servo[self.drogue_servo_channel].angle = 120.0
        print("Drogue deployed!")

    def buzzer(self, enabled: bool):
        """Drive the transistor buzzer circuit on GPIO16."""
        if self._gpio is not None:
            self._gpio.output(self.buzzer_gpio, self._gpio.HIGH if enabled else self._gpio.LOW)
