"""
Real hardware interface — Raspberry Pi 4 / GARUD HAT.

Confirmed hardware (2026-07-21):
  PCA9685 servo driver  — I2C (address in config/gains.yaml hardware.i2c_addresses.pca9685)
  EMAX ES3004 x2             — PCA9685 channels glider_left, glider_right   (brake servos)
  EMAX ES3004 x1             — PCA9685 channel  drogue_release              (parachute release)
  28BYJ-48 x1 (gimbal roll), MG90 x1 (gimbal pitch)               — PCA9685 channels gimbal_roll, gimbal_pitch   (camera gimbal)

Channel assignments are read from config/gains.yaml (section: servo_channels).
No hardcoded channel numbers anywhere in this file.

Install on Pi:  pip install adafruit-circuitpython-servokit
"""

from __future__ import annotations

import logging
import yaml

from hw_interface.base import HWInterface

logger = logging.getLogger(__name__)


def _load_servo_config():
    with open("config/gains.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    hw   = cfg.get("hardware", {})
    ch   = cfg.get("servo_channels", {})
    addr = hw.get("i2c_addresses", {}).get("pca9685", 0x40)
    return addr, ch


class RealHardware(HWInterface):
    """
    Hardware interface for the physical Raspberry Pi 4 / GARUD HAT.

    Servo angles are in degrees, range [60, 120] matching the SITL convention.
    Neutral = 90°.
    """

    def __init__(self):
        from adafruit_servokit import ServoKit

        pca_addr, self._ch = _load_servo_config()

        # 16-channel PCA9685 at the configured I2C address
        self._kit = ServoKit(channels=16, address=pca_addr)
        logger.info("PCA9685 ServoKit initialised at I2C 0x%02X", pca_addr)

        # Centre all servos at startup
        for ch in (
            self._ch.get("glider_left",    0),
            self._ch.get("glider_right",   1),
            self._ch.get("drogue_release", 2),
            self._ch.get("gimbal_roll",    3),
            self._ch.get("gimbal_pitch",   4),
        ):
            self._kit.servo[ch].angle = 90
        logger.info("All servos centred at 90°")

    # ------------------------------------------------------------------
    # Glider brake servos (EMAX ES3004 — channels 0 & 1)
    # ------------------------------------------------------------------

    def write_servos(self, left_deg: float, right_deg: float) -> None:
        """
        Command left and right brake servos.
        left_deg / right_deg are in degrees, clamped to [60, 120].
        """
        left_deg  = max(60.0, min(120.0, left_deg))
        right_deg = max(60.0, min(120.0, right_deg))
        self._kit.servo[self._ch.get("glider_left",  0)].angle = left_deg
        self._kit.servo[self._ch.get("glider_right", 1)].angle = right_deg

    # ------------------------------------------------------------------
    # Drogue release servo (EMAX ES3004 — channel 2)
    # ------------------------------------------------------------------

    def trigger_drogue(self) -> None:
        """
        Commands the drogue release servo to the open position (120°).
        This should only be called once by the state machine at 600m AGL.
        """
        release_ch = self._ch.get("drogue_release", 2)
        self._kit.servo[release_ch].angle = 120
        logger.warning("[HW] DROGUE RELEASE commanded on PCA9685 ch %d", release_ch)

    # ------------------------------------------------------------------
    # Gimbal servos (28BYJ-48/MG90 — channels 3 & 4)
    # ------------------------------------------------------------------

    def write_gimbal(self, roll_deg: float, pitch_deg: float) -> None:
        """
        Command the camera gimbal.
        roll_deg / pitch_deg are in degrees, clamped to [-45, 45] then offset to servo range.
        """
        roll_deg  = max(-45.0, min(45.0, roll_deg))
        pitch_deg = max(-45.0, min(45.0, pitch_deg))
        # Map -45…+45° → 45…135° servo angle
        self._kit.servo[self._ch.get("gimbal_roll",  3)].angle = roll_deg  + 90.0
        self._kit.servo[self._ch.get("gimbal_pitch", 4)].angle = pitch_deg + 90.0

    # ------------------------------------------------------------------
    # HWInterface abstract method stubs (sensors live in sensors/drivers.py)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        logger.info("RealHardware.initialize() complete.")

    def read_imu(self):
        raise NotImplementedError("Call BNO085 driver directly from flight_computer.py")

    def read_baro(self):
        raise NotImplementedError("Call BMP388 driver directly from flight_computer.py")

    def read_gps(self):
        raise NotImplementedError("Call GPS driver directly from flight_computer.py")

    def read_power(self):
        raise NotImplementedError("Call INA219 driver directly from flight_computer.py")

