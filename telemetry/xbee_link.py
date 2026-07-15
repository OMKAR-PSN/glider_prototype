import serial
from typing import Optional
from .packet import TelemetryPacket

class XBeeLink:
    """
    Serial transport class for XBee telemetry.
    """
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial: Optional[serial.Serial] = None

    def connect(self):
        """Connect to the XBee serial port."""
        try:
            self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        except serial.SerialException as e:
            print(f"Failed to connect to XBee on {self.port}: {e}")
            self._serial = None

    def disconnect(self):
        """Disconnect from the XBee serial port."""
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
