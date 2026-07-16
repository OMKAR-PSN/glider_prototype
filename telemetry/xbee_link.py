from typing import Optional
from hw_interface.pinmap import LORA_UART
from .packet import TelemetryPacket

try:
    import serial
except ImportError:
    serial = None

class LoRaLink:
    """
    Serial transport class for the UART LoRa telemetry module on connector J3.

    Schema_Draft_2.pdf routes:
    - TX_Lora to GPIO14/TXD0
    - RX_Lora to GPIO15/RXD0
    - M0 to GPIO23
    - M1 to GPIO24
    - AUX to GPIO25
    """
    def __init__(self, port: str = LORA_UART.port, baudrate: int = LORA_UART.baudrate,
                 m0_gpio: int | None = LORA_UART.m0_gpio,
                 m1_gpio: int | None = LORA_UART.m1_gpio,
                 aux_gpio: int | None = LORA_UART.aux_gpio):
        self.port = port
        self.baudrate = baudrate
        self.m0_gpio = m0_gpio
        self.m1_gpio = m1_gpio
        self.aux_gpio = aux_gpio
        self._serial = None

    def connect(self):
        """Connect to the LoRa serial port."""
        if serial is None:
            print("pyserial is not installed; LoRa telemetry disabled.")
            self._serial = None
            return

        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        except serial.SerialException as e:
            print(f"Failed to connect to LoRa telemetry on {self.port}: {e}")
            self._serial = None

    def disconnect(self):
        """Disconnect from the LoRa serial port."""
        if self._serial and self._serial.is_open:
            self._serial.close()

    def send_packet(self, packet: TelemetryPacket) -> bool:
        """
        Sends a telemetry packet over the XBee link.
        """
        if not self._serial or not self._serial.is_open:
            return False
        
        csv_line = packet.to_csv_line() + "\n"
        try:
            self._serial.write(csv_line.encode('utf-8'))
            return True
        except serial.SerialException:
            return False

    def receive_packet(self) -> Optional[TelemetryPacket]:
        """
        Receives a telemetry packet from the XBee link.
        """
        if not self._serial or not self._serial.is_open:
            return None
            
        try:
            if self._serial.in_waiting > 0:
                line = self._serial.readline().decode('utf-8').strip()
                if line:
                    return TelemetryPacket.from_csv_line(line)
        except (serial.SerialException, ValueError) as e:
            print(f"Error receiving/parsing packet: {e}")
            
        return None

XBeeLink = LoRaLink
