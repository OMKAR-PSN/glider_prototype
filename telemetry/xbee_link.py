"""
telemetry/xbee_link.py -- XBee 2.4 GHz telemetry driver (ACTIVE)

Hardware:
  Module   : XBee 802.15.4 (S2C or compatible)
  Interface: UART /dev/ttyAMA0 (GPIO14=RX, GPIO15=TX)
             GPS is on SPI so ttyAMA0 is now free for XBee exclusively.

  Prerequisite: Bluetooth must be disabled so ttyAMA0 is free.
    /boot/config.txt: dtoverlay=disable-bt
    sudo systemctl disable hciuart

IN-SPACe Team 002 assignment (2026-27 competition):
  PAN ID:         0x1001
  Main channel:   Ch 11 (2405 MHz)   -- use by default
  Backup channel: Ch 21 (2455 MHz)   -- switch ONLY on Mission Control call
  Mode:           IEEE 802.15.4 (NOT XBee 900 HP)

COMPETITION RULES:
  - Use MAIN channel by default.
  - Switch to BACKUP only when Mission Control explicitly instructs you.
  - NEVER transmit on any channel not listed above.
  - Auto-shutoff: transmitter silenced 90 min after launch detection.

Hardware note: PAN ID and channel must be pre-programmed into the XBee module
using XCTU before flight. Switching channels during flight uses AT commands (ATCH).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import serial

from .packet import TelemetryPacket

logger = logging.getLogger(__name__)


# Team 002 assigned channels
XBEE_CHANNEL_MAIN   = 0x0B    # Ch 11 = 2405 MHz
XBEE_CHANNEL_BACKUP = 0x15    # Ch 21 = 2455 MHz
XBEE_PAN_ID         = 0x1001  # Team 002

# Auto-shutoff: 90 minutes after launch detection (competition rule)
SHUTOFF_SECONDS = 90 * 60


class XBeeLink:
    """
    Serial transport for the 2.4 GHz XBee telemetry link.

    Usage:
        link = XBeeLink(port="/dev/ttyUSB1")
        link.connect()
        link.send_packet(packet)

    To switch to backup channel (Mission Control instruction only):
        link.switch_to_backup()
    """

    def __init__(self, port: str = "/dev/ttyUSB1", baudrate: int = 9600):
        self.port     = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None
        self._launch_time: Optional[float] = None
        self._silenced = False
        self._active_channel = XBEE_CHANNEL_MAIN

    def connect(self) -> bool:
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
            logger.info("XBee connected on %s | PAN 0x%04X | Ch 0x%02X (MAIN)",
                        self.port, XBEE_PAN_ID, self._active_channel)
            return True
        except serial.SerialException as e:
            logger.error("XBee connection failed on %s: %s", self.port, e)
            self._serial = None
            return False

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("XBee disconnected.")

    def notify_launch(self) -> None:
        """
        Call this when launch is detected.
        Starts the 90-minute competition auto-shutoff countdown.
        """
        self._launch_time = time.time()
        logger.info("XBee: launch detected — auto-shutoff in %.0f minutes.", SHUTOFF_SECONDS / 60)

    def _check_shutoff(self) -> bool:
        """Returns True if the transmitter must be silenced."""
        if self._silenced:
            return True
        if self._launch_time and (time.time() - self._launch_time) >= SHUTOFF_SECONDS:
            self._silenced = True
            logger.warning(
                "XBee: 90-MINUTE AUTO-SHUTOFF triggered (competition rule). "
                "Transmitter silenced. Do NOT re-enable without Mission Control clearance."
            )
        return self._silenced

    def switch_to_backup(self) -> bool:
        """
        Switches to the backup channel (Ch 21 / 2455 MHz).
        Call ONLY when Mission Control explicitly instructs you to.
        Sends AT command sequence to the XBee module.
        """
        if not self._serial or not self._serial.is_open:
            logger.error("XBee: cannot switch channel — not connected.")
            return False
        try:
            # Enter AT command mode
            time.sleep(1.1)
            self._serial.write(b"+++")
            time.sleep(1.1)
            response = self._serial.read(3)
            if b"OK" not in response:
                logger.error("XBee: AT command mode entry failed.")
                return False
            # Set channel to backup
            self._serial.write(f"ATCH{XBEE_CHANNEL_BACKUP:02X}\r".encode())
            time.sleep(0.1)
            # Apply and exit
            self._serial.write(b"ATAC\r")
            time.sleep(0.1)
            self._serial.write(b"ATCN\r")
            self._active_channel = XBEE_CHANNEL_BACKUP
            logger.warning("XBee: switched to BACKUP channel 0x%02X (2455 MHz).", XBEE_CHANNEL_BACKUP)
            return True
        except serial.SerialException as e:
            logger.error("XBee channel switch failed: %s", e)
            return False

    def switch_to_main(self) -> bool:
        """Switch back to Main channel Ch 11 (2405 MHz)."""
        if not self._serial or not self._serial.is_open:
            return False
        try:
            time.sleep(1.1)
            self._serial.write(b"+++")
            time.sleep(1.1)
            self._serial.write(f"ATCH{XBEE_CHANNEL_MAIN:02X}\r".encode())
            time.sleep(0.1)
            self._serial.write(b"ATAC\r")
            time.sleep(0.1)
            self._serial.write(b"ATCN\r")
            self._active_channel = XBEE_CHANNEL_MAIN
            logger.info("XBee: switched back to MAIN channel 0x%02X (2405 MHz).", XBEE_CHANNEL_MAIN)
            return True
        except serial.SerialException as e:
            logger.error("XBee main channel restore failed: %s", e)
            return False

    def send_packet(self, packet: TelemetryPacket) -> bool:
        """Sends a telemetry packet. Silently drops if shutoff has been triggered."""
        if self._check_shutoff():
            return False
        if not self._serial or not self._serial.is_open:
            return False
        try:
            csv_line = packet.to_csv_line() + "\n"
            self._serial.write(csv_line.encode("utf-8"))
            return True
        except serial.SerialException as e:
            logger.error("XBee send error: %s", e)
            return False

    def receive_packet(self) -> Optional[TelemetryPacket]:
        """Receives and parses an incoming telemetry packet (telecommand responses)."""
        if not self._serial or not self._serial.is_open:
            return None
        try:
            if self._serial.in_waiting > 0:
                line = self._serial.readline().decode("utf-8").strip()
                if line:
                    return TelemetryPacket.from_csv_line(line)
        except (serial.SerialException, ValueError) as e:
            logger.error("XBee receive error: %s", e)
        return None
