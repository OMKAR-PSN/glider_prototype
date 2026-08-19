"""
LoRa E22 telemetry driver — GARUD HAT schematic confirmed.

Hardware:
  Module:   LoRa E22 (UART interface)
  Connector: J3 — schematic-confirmed pin routing:
      Pin 3 (RX_Lora)  → GPIO15 / Pi Pin 10 → /dev/ttyAMA0 RX
      Pin 4 (TX_Lora)  → GPIO14 / Pi Pin 8  → /dev/ttyAMA0 TX
      Pin 5 (AUX_Lora) → GPIO18 / Pin 12    → NOT routed to Pi GPIO (floating)

  M0 and M1: hardwired to GND on the PCB — module is permanently in NORMAL
  transmission mode. No GPIO control needed or possible.

  AUX pin: not connected to Pi GPIO. Driver uses a fixed 10ms delay instead
  of polling AUX for module-ready state.

  CRITICAL: GPS NEO-M8N is also on GPIO14/15 (/dev/ttyAMA0).
  GPS and LoRa CANNOT run simultaneously on the same UART.
  Resolution required before flight — see config/gains.yaml CONFLICT note.

IN-SPACe Team 002 frequency assignment:
  Main:    865.50 MHz
  Backup:  867.25 MHz
  SF:      7
  BW:      125 kHz
  CR:      4/5
  Sync:    0x81
  TX pad:  +10 dBm   TX flight: +14 dBm

COMPETITION RULES:
  - MAIN channel by default. Switch to BACKUP only on Mission Control call.
  - Auto-shutoff: both transmitters silenced 90 min after launch detection.
  - FORBIDDEN bands: 433 MHz (rocket ignition), 446 MHz (voice).
  - Never transmit on frequencies not assigned above.

The E22 operates in NORMAL mode (M0=LOW, M1=LOW).
AUX pin is HIGH when the module is idle and ready to accept data.
The module must be pre-configured with the correct frequency, SF, BW, and
sync word using LLCC68 AT commands or the E22 PC configuration tool BEFORE
flight. This driver assumes the module is already configured.

Install: pip install pyserial RPi.GPIO
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import serial

from .packet import TelemetryPacket

logger = logging.getLogger(__name__)

# Auto-shutoff: 90 minutes after launch detection (competition rule)
SHUTOFF_SECONDS = 90 * 60


class LoRaLink:
    """
    Serial transport for the LoRa E22 telemetry link.

    Usage:
        link = LoRaLink(port="/dev/ttyAMA1", aux_pin=None)
        link.connect()
        link.notify_launch()
        link.send_packet(packet)

    To switch to backup frequency (Mission Control only):
        link.switch_to_backup()
    """

    def __init__(
        self,
        port: str    = "/dev/ttyAMA0",
        baudrate: int = 9600,
    ):
        """
        M0 and M1 are hardwired GND on the PCB — NORMAL mode is permanent.
        AUX is not routed to any GPIO — fixed delay used instead.
        """
        self._port      = port
        self._baudrate  = baudrate
        self._serial: Optional[serial.Serial] = None
        self._launch_time: Optional[float]    = None
        self._silenced  = False
        self._on_backup = False

    def connect(self) -> bool:
        """Open the serial port. M0/M1 already LOW (hardwired), no GPIO setup needed."""
        try:
            self._serial = serial.Serial(self._port, self._baudrate, timeout=1.0)
            logger.info(
                "LoRaLink connected on %s @ %d baud | 865.50 MHz SF7 Sync=0x81 | M0/M1 hardwired LOW",
                self._port, self._baudrate
            )
            return True
        except Exception as e:
            logger.error("LoRaLink connection failed on %s: %s", self._port, e)
            self._serial = None
            return False

    def disconnect(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("LoRaLink disconnected.")

    def notify_launch(self) -> None:
        """
        Call when launch is detected by the state machine.
        Starts the 90-minute competition auto-shutoff countdown.
        """
        self._launch_time = time.time()
        logger.info("LoRa: launch detected — auto-shutoff in %.0f min.", SHUTOFF_SECONDS / 60)

    def _check_shutoff(self) -> bool:
        """Returns True if the transmitter must be silenced (90 min rule)."""
        if self._silenced:
            return True
        if self._launch_time and (time.time() - self._launch_time) >= SHUTOFF_SECONDS:
            self._silenced = True
            logger.warning(
                "LoRa: 90-MINUTE AUTO-SHUTOFF triggered (IN-SPACe rule). "
                "Transmitter silenced. Do NOT re-enable without Mission Control clearance."
            )
        return self._silenced

    def _wait_aux_ready(self, timeout_s: float = 0.5) -> None:
        """
        AUX_Lora is NOT connected to any Pi GPIO on this PCB revision.
        Fixed 10ms delay used instead of AUX polling.
        If AUX is wired in a future PCB revision, add GPIO.input() polling here.
        """
        time.sleep(0.010)   # 10ms fixed delay — sufficient for E22 at 9600 baud

    def switch_to_backup(self) -> bool:
        """
        Logs that a backup frequency switch is needed.
        The E22 frequency is set in hardware configuration (not via UART in
        normal mode). Switching frequency mid-flight requires entering config
        mode (M0=LOW, M1=HIGH) and sending AT command — complex and risky.

        For now: log the instruction for the operator and note in telemetry.
        Full AT-command switching can be added if required by the competition.
        """
        self._on_backup = True
        logger.warning(
            "LoRa: BACKUP frequency requested (867.25 MHz). "
            "Manual reconfiguration required via M1=HIGH + AT+FREQ command. "
            "Contact Mission Control to confirm switch procedure."
        )
        return False   # not yet automated

    def send_packet(self, packet: TelemetryPacket) -> bool:
        """
        Sends a telemetry packet over LoRa.
        Silently drops if 90-minute shutoff has triggered.
        """
        if self._check_shutoff():
            return False
        if not self._serial or not self._serial.is_open:
            return False
        try:
            self._wait_aux_ready()
            csv_line = packet.to_csv_line() + "\n"
            self._serial.write(csv_line.encode("utf-8"))
            return True
        except serial.SerialException as e:
            logger.error("LoRa send error: %s", e)
            return False

    def receive_packet(self) -> Optional[TelemetryPacket]:
        """Receives an incoming packet (telecommands from ground station)."""
        if not self._serial or not self._serial.is_open:
            return None
        try:
            if self._serial.in_waiting > 0:
                line = self._serial.readline().decode("utf-8", errors="replace").strip()
                if line:
                    return TelemetryPacket.from_csv_line(line)
        except (serial.SerialException, ValueError) as e:
            logger.error("LoRa receive error: %s", e)
        return None
