from dataclasses import dataclass
from typing import Optional

@dataclass
class TelemetryPacket:
    """
    CanSat Telemetry Packet Schema.
    """
    frame_id: int
    time_stamp: float
    gnss_latitude: float
    gnss_longitude: float
    gnss_altitude: float
    baro_altitude: float
    roll: float
    pitch: float
    yaw: float
    gimbal_roll: float
    gimbal_pitch: float
    camera_status: int
    terrain_process_status: int
    image_filename: str

    def to_csv_line(self) -> str:
        """Serializes the packet to a CSV line."""
        fields = [
            str(self.frame_id),
            f"{self.time_stamp:.3f}",
            f"{self.gnss_latitude:.6f}",
            f"{self.gnss_longitude:.6f}",
            f"{self.gnss_altitude:.2f}",
            f"{self.baro_altitude:.2f}",
            f"{self.roll:.2f}",
            f"{self.pitch:.2f}",
            f"{self.yaw:.2f}",
            f"{self.gimbal_roll:.2f}",
            f"{self.gimbal_pitch:.2f}",
            str(self.camera_status),
            str(self.terrain_process_status),
            self.image_filename
        ]
        return ",".join(fields)

    @classmethod
    def from_csv_line(cls, line: str) -> "TelemetryPacket":
        """Deserializes a CSV line into a TelemetryPacket."""
        parts = line.strip().split(",")
        if len(parts) != 14:
            raise ValueError(f"Expected 14 fields, got {len(parts)} in CSV line")
        
        return cls(
            frame_id=int(parts[0]),
            time_stamp=float(parts[1]),
            gnss_latitude=float(parts[2]),
            gnss_longitude=float(parts[3]),
            gnss_altitude=float(parts[4]),
            baro_altitude=float(parts[5]),
            roll=float(parts[6]),
            pitch=float(parts[7]),
            yaw=float(parts[8]),
            gimbal_roll=float(parts[9]),
            gimbal_pitch=float(parts[10]),
            camera_status=int(parts[11]),
            terrain_process_status=int(parts[12]),
            image_filename=parts[13]
        )
