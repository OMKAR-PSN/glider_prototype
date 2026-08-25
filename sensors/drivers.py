"""
sensors/drivers.py — GARUD HAT hardware drivers (schematic confirmed Rev 2)

Hardware map (all confirmed from GARUD HAT Rev 2 schematic):
  BNO085   : SPI0, CS=GPIO5 (D5), RST=GPIO6 (D6), INT=GPIO27 (D27)
  BMP388   : SPI0, CS=GPIO22 (D22), INT=GPIO17 (D17)  [shared SPI bus]
  INA219   : I2C, address 0x41 (ADDR pin to VCC, schematic U4)
  PCA9685  : I2C, address 0x40 (A0-A5 all to GND), OE=GPIO4
  GPS      : UART /dev/ttyAMA0 (GPIO14=TX, GPIO15=RX)
  XBee     : UART (shared /dev/ttyAMA0 or ttyUSB)
  Buzzer   : GPIO16 -> 220 ohm -> 2N2219 transistor -> BZ1

Install requirements (run once on the Pi):
  pip install adafruit-circuitpython-bno08x
  pip install adafruit-circuitpython-bmp3xx
  pip install adafruit-circuitpython-ina219
  pip install pyserial pynmea2 RPi.GPIO
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from typing import Optional

from utils.helpers import HardwareError   # raises on sensor failure

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IMUData:
    """Fused attitude + raw inertial data from BNO085."""
    roll:               float
    pitch:              float
    yaw:                float
    gyro_p:             float
    gyro_q:             float
    gyro_r:             float
    accel_x:            float
    accel_y:            float
    accel_z:            float
    mag_x:              float
    mag_y:              float
    mag_z:              float
    calibration_status: int = 0   # BNO085 calibration level (0=uncal, 3=fully cal)


@dataclass
class BaroData:
    """Barometric reading from BMP388."""
    pressure:    float   # hPa
    temperature: float   # deg C
    altitude:    float   # metres AGL (zeroed at init)


@dataclass
class GPSData:
    """Parsed NMEA fix from NEO-M8N."""
    latitude:     float   # decimal degrees
    longitude:    float   # decimal degrees
    altitude:     float   # metres MSL
    ground_speed: float   # m/s
    heading:      float   # radians, course-over-ground
    fix:          bool


@dataclass
class PowerData:
    """Voltage and current from INA219."""
    voltage: float   # V
    current: float   # A


# ---------------------------------------------------------------------------
# BNO085 -- Attitude / IMU (SPI0, GARUD HAT Rev 2)
# ---------------------------------------------------------------------------

class BNO085:
    """
    AHRS driver for the BNO085 over SPI (GARUD HAT Rev 2).

    Library  : adafruit-circuitpython-bno08x
    Interface: SPI0 shared bus
      CS  = GPIO5  (board.D5,  Pi pin 29)
      RST = GPIO6  (board.D6,  Pi pin 31)
      INT = GPIO27 (board.D27, Pi pin 13) -- used for interrupt-driven reads

    Install:
      pip install adafruit-circuitpython-bno08x

    Stability classification strings:
        "On Table" | "Stationary" | "Stable" | "In Motion"
    """

    _STABLE_STATES = {"On Table", "Stationary", "Stable"}

    def __init__(self) -> None:
        try:
            import board
            import busio
            import digitalio
            from adafruit_bno08x import (
                BNO_REPORT_ROTATION_VECTOR,
                BNO_REPORT_GYROSCOPE,
                BNO_REPORT_LINEAR_ACCELERATION,
                BNO_REPORT_MAGNETOMETER,
            )
            from adafruit_bno08x.spi import BNO08X_SPI

            spi       = busio.SPI(board.SCK, board.MOSI, board.MISO)
            cs        = digitalio.DigitalInOut(board.D5)    # CS_BNO  = GPIO5
            interrupt = digitalio.DigitalInOut(board.D27)   # INT_BNO = GPIO27
            reset     = digitalio.DigitalInOut(board.D6)    # RST_BNO = GPIO6
            self._bno = BNO08X_SPI(spi, cs, interrupt, reset)  # same as GARUD repo
            self._bno.enable_feature(BNO_REPORT_ROTATION_VECTOR)
            self._bno.enable_feature(BNO_REPORT_GYROSCOPE)
            self._bno.enable_feature(BNO_REPORT_LINEAR_ACCELERATION)
            self._bno.enable_feature(BNO_REPORT_MAGNETOMETER)
            logger.info("BNO085 initialised over SPI (CS=GPIO5, INT=GPIO27, RST=GPIO6).")
        except Exception as exc:
            raise HardwareError(
                f"BNO085 not detected on SPI (CS=GPIO5, INT=GPIO27, RST=GPIO6): {exc}"
            ) from exc

    def calibration_ok(self, min_level: int = 2) -> bool:
        """
        Returns True when the sensor stability is in a known-good state.
        Uses stability_classification (only exposed quality metric in this library).
        """
        try:
            return self._bno.stability_classification in self._STABLE_STATES
        except Exception:
            return False

    def read(self) -> Optional[IMUData]:
        try:
            qi, qj, qk, qr = self._bno.quaternion

            # Quaternion -> Euler (ZYX convention, radians)
            sinr_cosp = 2.0 * (qr * qi + qj * qk)
            cosr_cosp = 1.0 - 2.0 * (qi * qi + qj * qj)
            roll  = math.atan2(sinr_cosp, cosr_cosp)

            sinp  = max(-1.0, min(1.0, 2.0 * (qr * qj - qk * qi)))
            pitch = math.asin(sinp)

            siny_cosp = 2.0 * (qr * qk + qi * qj)
            cosy_cosp = 1.0 - 2.0 * (qj * qj + qk * qk)
            yaw   = math.atan2(siny_cosp, cosy_cosp)

            gx, gy, gz = self._bno.gyro
            ax, ay, az = self._bno.linear_acceleration
            mx, my, mz = self._bno.magnetic
            cal_status = getattr(self._bno, "calibration_status", 0) or 0

            return IMUData(
                roll=roll, pitch=pitch, yaw=yaw,
                gyro_p=gx, gyro_q=gy, gyro_r=gz,
                accel_x=ax, accel_y=ay, accel_z=az,
                mag_x=mx, mag_y=my, mag_z=mz,
                calibration_status=cal_status,
            )
        except Exception as exc:
            raise HardwareError(f"BNO085 read failed: {exc}") from exc


# ---------------------------------------------------------------------------
# BMP388 -- Barometer (SPI0, CS = GPIO22, GARUD HAT Rev 2)
# ---------------------------------------------------------------------------

class BMP388:
    """
    Barometric altitude driver for the BMP388 over SPI (GARUD HAT Rev 2).

    Library  : adafruit-circuitpython-bmp3xx
    Interface: SPI0 shared bus
      CS  = GPIO22 (board.D22, Pi pin 15)
      INT = GPIO17 (board.D17, Pi pin 11) -- not polled, interrupt available

    NOTE: BMP388 and BNO085 share the same SPI bus (GPIO11/10/9).
          CS lines are separate — only one is asserted at a time.

    Install:
      pip install adafruit-circuitpython-bmp3xx

    AGL altitude is zeroed at object construction (launch-pad level).
    """

    def __init__(self, cs_pin_name: str = "D22") -> None:
        try:
            import board
            import busio
            import digitalio
            import adafruit_bmp3xx

            spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
            cs  = digitalio.DigitalInOut(getattr(board, cs_pin_name))
            self._bmp = adafruit_bmp3xx.BMP3XX_SPI(spi, cs)
            self._bmp.pressure_oversampling    = 8
            self._bmp.temperature_oversampling = 2
            self._ground_alt = self._bmp.altitude
            logger.info(
                "BMP388 (SPI, CS=%s/GPIO22) initialised. Ground ref %.1f m MSL.",
                cs_pin_name, self._ground_alt,
            )
        except Exception as exc:
            raise HardwareError(
                f"BMP388 not detected on SPI (CS={cs_pin_name}/GPIO22): {exc}"
            ) from exc

    def read(self) -> Optional[BaroData]:
        try:
            return BaroData(
                pressure    = self._bmp.pressure,
                temperature = self._bmp.temperature,
                altitude    = self._bmp.altitude - self._ground_alt,
            )
        except Exception as exc:
            raise HardwareError(f"BMP388 read failed: {exc}") from exc


# ---------------------------------------------------------------------------
# INA219 -- Power Monitor (I2C, address 0x41)
# ---------------------------------------------------------------------------

class INA219:
    """
    Voltage and current monitor.

    Library  : adafruit-circuitpython-ina219
    I2C addr : 0x41 -- ADDR pin pulled to VCC on schematic U4.
               Default 0x40 is taken by PCA9685; conflict resolved by
               jumpering INA219 ADDR to VCC -> 0x41.
    """

    def __init__(self, address: int = 0x41) -> None:
        try:
            import board
            import busio
            from adafruit_ina219 import INA219 as _INA219

            i2c = busio.I2C(board.SCL, board.SDA)
            self._ina = _INA219(i2c, addr=address)
            self._simulated = False
            logger.info("INA219 initialised at I2C 0x%02X.", address)
        except Exception as e:
            logger.warning("INA219 init failed (%s) -- using simulated data.", e)
            self._simulated = True

    def read(self) -> Optional[PowerData]:
        if self._simulated:
            return PowerData(voltage=5.0, current=0.5)
        try:
            return PowerData(
                voltage=self._ina.bus_voltage + self._ina.shunt_voltage / 1000.0,
                current=self._ina.current / 1000.0,   # mA -> A
            )
        except Exception as e:
            logger.error("INA219 read error: %s", e)
            return None


# ---------------------------------------------------------------------------
# GPS -- NEO-M8N (UART /dev/ttyAMA0)
# ---------------------------------------------------------------------------

class GPS:
    """
    SPI driver for the u-blox NEO-M8N GPS module.

    Interface: SPI0, CS = GPIO7 (CE1)
               BMP388 is on CE0 (GPIO8) -- both share the same SPI bus.

    IMPORTANT: The NEO-M8N defaults to UART mode.
    To enable SPI mode the D_SEL pin MUST be pulled LOW on the PCB.
    Confirm this has been done on the GARUD HAT before testing.

    SPI protocol: poll by sending 0xFF dummy bytes. The module returns NMEA
    sentence bytes when it has data, or 0xFF when idle. This driver accumulates
    bytes into a buffer and parses complete NMEA sentences.

    SPI clock speed: max 5.4 MHz for NEO-M8N. We use 1 MHz to be safe.
    """

    def __init__(self, cs_pin_name: str = "CE1") -> None:
        try:
            import board
            import busio
            import digitalio

            spi = busio.SPI(board.SCK, board.MOSI, board.MISO)
            cs  = digitalio.DigitalInOut(getattr(board, cs_pin_name))
            cs.direction = digitalio.Direction.OUTPUT
            cs.value = True   # deassert CS (active low)

            self._spi        = spi
            self._cs         = cs
            self._rx_buffer  = bytearray()
            self._pynmea2    = __import__("pynmea2")
            self._simulated  = False
            self._last_fix: Optional[GPSData] = None
            logger.info("GPS NEO-M8N (SPI, CS=%s) initialised.", cs_pin_name)
        except Exception as e:
            logger.warning("GPS SPI init failed (%s) -- using simulated data.", e)
            self._simulated = True

    def _poll_spi(self, num_bytes: int = 128) -> bytes:
        """
        Transfer num_bytes of 0xFF dummy bytes to the module.
        Returns only the non-0xFF bytes received (actual GPS data).
        """
        tx = bytes([0xFF] * num_bytes)
        rx = bytearray(num_bytes)
        while not self._spi.try_lock():
            pass
        try:
            self._spi.configure(baudrate=1_000_000, polarity=0, phase=0)
            self._cs.value = False
            self._spi.write_readinto(tx, rx)
            self._cs.value = True
        finally:
            self._spi.unlock()
        return bytes(b for b in rx if b != 0xFF)

    def read(self) -> Optional[GPSData]:
        if self._simulated:
            return GPSData(
                latitude=18.5204, longitude=73.8567,
                altitude=0.0, ground_speed=0.0, heading=0.0, fix=False,
            )
        try:
            self._rx_buffer += self._poll_spi(128)

            # Parse all complete NMEA sentences in the buffer
            result: Optional[GPSData] = None
            while b'\n' in self._rx_buffer:
                raw_line, self._rx_buffer = self._rx_buffer.split(b'\n', 1)
                sentence = raw_line.decode('ascii', errors='replace').strip()
                if not sentence.startswith('$'):
                    continue
                try:
                    msg = self._pynmea2.parse(sentence)

                    # GGA -- position and fix quality
                    if isinstance(msg, self._pynmea2.types.talker.GGA):
                        if msg.gps_qual == 0:
                            result = GPSData(0.0, 0.0, 0.0, 0.0, 0.0, False)
                        else:
                            spd = self._last_fix.ground_speed if self._last_fix else 0.0
                            hdg = self._last_fix.heading      if self._last_fix else 0.0
                            result = GPSData(
                                latitude=msg.latitude,
                                longitude=msg.longitude,
                                altitude=float(msg.altitude or 0.0),
                                ground_speed=spd,
                                heading=hdg,
                                fix=True,
                            )
                            self._last_fix = result

                    # VTG -- speed and course-over-ground
                    elif isinstance(msg, self._pynmea2.types.talker.VTG):
                        spd_kmh = float(msg.spd_over_grnd_kmph or 0.0)
                        course  = float(msg.true_track or 0.0)
                        if self._last_fix:
                            result = GPSData(
                                latitude=self._last_fix.latitude,
                                longitude=self._last_fix.longitude,
                                altitude=self._last_fix.altitude,
                                ground_speed=spd_kmh / 3.6,
                                heading=math.radians(course),
                                fix=self._last_fix.fix,
                            )
                            self._last_fix = result
                except Exception:
                    pass

            return result
        except Exception as e:
            logger.error("GPS SPI read error: %s", e)
            return None


# ---------------------------------------------------------------------------
# BuzzerDriver -- Active buzzer on GPIO16
# ---------------------------------------------------------------------------

class BuzzerDriver:
    """
    Active piezo buzzer driver.

    Schematic: GPIO16 -> 220 ohm resistor -> base of 2N2219 NPN transistor -> BZ1.
    Driving GPIO16 HIGH energises the buzzer.

    Compatibility:
      Pi 4 : uses RPi.GPIO
      Pi 5 : uses lgpio  (RPi.GPIO does not support Pi 5 RP1 chip)

    Beep patterns:
      arm_confirmation     -- 3 short beeps (system armed, all sensors green)
      recovery_beacon      -- 1 long + 2 short (glider on ground, find me)
      gps_dropout_warning  -- 1 long beep     (GPS stale, AGC fallback active)
      agc_fallback_warning -- 2 short beeps   (RL watchdog fired)
    """

    def __init__(self, gpio_pin: int = 16) -> None:
        self._pin = gpio_pin
        self._mode = None  # 'lgpio' | 'rpigpio' | None (disabled)

        # Try lgpio first (Pi 5)
        try:
            import lgpio
            self._h = lgpio.gpiochip_open(0)
            lgpio.gpio_claim_output(self._h, gpio_pin, 0)
            self._lgpio = lgpio
            self._mode = "lgpio"
            logger.info("BuzzerDriver (lgpio, Pi 5) initialised on GPIO%d.", gpio_pin)
            return
        except Exception:
            pass

        # Fallback: RPi.GPIO (Pi 4)
        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(gpio_pin, GPIO.OUT, initial=GPIO.LOW)
            self._mode = "rpigpio"
            logger.info("BuzzerDriver (RPi.GPIO, Pi 4) initialised on GPIO%d.", gpio_pin)
            return
        except Exception:
            pass

        logger.warning("BuzzerDriver: no GPIO library available -- buzzer disabled.")

    def _beep(self, on_s: float, off_s: float = 0.1, count: int = 1) -> None:
        if self._mode is None:
            return
        for _ in range(count):
            self._gpio_set(1)
            time.sleep(on_s)
            self._gpio_set(0)
            time.sleep(off_s)

    def _gpio_set(self, value: int) -> None:
        if self._mode == "lgpio":
            self._lgpio.gpio_write(self._h, self._pin, value)
        elif self._mode == "rpigpio":
            self._GPIO.output(self._pin, value)

    def arm_confirmation(self) -> None:
        """3 short beeps -- system armed, all sensors green."""
        self._beep(on_s=0.10, off_s=0.10, count=3)

    def recovery_beacon(self) -> None:
        """1 long + 2 short beeps -- glider on ground, locate me."""
        self._beep(on_s=0.60, off_s=0.15, count=1)
        self._beep(on_s=0.15, off_s=0.15, count=2)

    def gps_dropout_warning(self) -> None:
        """1 long beep -- GPS is stale, AGC fallback engaged."""
        self._beep(on_s=0.80, off_s=0.10, count=1)

    def agc_fallback_warning(self) -> None:
        """2 short beeps -- RL watchdog fired, falling back to AGC."""
        self._beep(on_s=0.15, off_s=0.15, count=2)

    def cleanup(self) -> None:
        """Release GPIO resources."""
        if self._mode == "lgpio":
            try:
                self._lgpio.gpiochip_close(self._h)
            except Exception:
                pass
        elif self._mode == "rpigpio":
            try:
                self._GPIO.cleanup(self._pin)
            except Exception:
                pass

