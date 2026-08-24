"""BNO085 SPI IMU hardware test — adapted from TARSR/GARUD raspi_test_code.

Usage (on Pi):
    python -m tests.test_bno085_hw             # quick one-shot check
    python -m tests.test_bno085_hw --stream    # continuous stream at 50 Hz
"""

from __future__ import annotations

import argparse
import time

from sensors.drivers import BNO085, IMUData
from utils.helpers import HardwareError
from utils.logger import build_logger

log = build_logger(name="test_bno085", save_logs=True)


def _format(d: IMUData) -> str:
    return (
        f"Roll:   {d.roll:+.3f} rad\n"
        f"Pitch:  {d.pitch:+.3f} rad\n"
        f"Yaw:    {d.yaw:+.3f} rad\n"
        f"Gyro:   ({d.gyro_p:+.3f}, {d.gyro_q:+.3f}, {d.gyro_r:+.3f}) rad/s\n"
        f"Accel:  ({d.accel_x:+.3f}, {d.accel_y:+.3f}, {d.accel_z:+.3f}) m/s²\n"
        f"Mag:    ({d.mag_x:+.2f}, {d.mag_y:+.2f}, {d.mag_z:+.2f}) uT\n"
        f"CalSts: {d.calibration_status}"
    )


def quick_check() -> bool:
    """Read one BNO085 sample — PASS/FAIL."""
    log.info("BNO085 quick check starting...")
    try:
        sensor = BNO085()
        data = sensor.read()
        log.success(f"BNO085 OK  |  Cal={data.calibration_status}  "
                    f"|  Yaw={data.yaw:+.3f} rad  "
                    f"|  Accel_z={data.accel_z:+.2f} m/s²")
        return True
    except HardwareError as exc:
        log.error(f"BNO085 FAIL: {exc}")
        return False


def calibration_wait(timeout_s: float = 60.0) -> bool:
    """Wait until BNO085 reports a stable calibration state."""
    log.info(f"Waiting for BNO085 calibration (timeout={timeout_s:.0f}s)...")
    try:
        sensor = BNO085()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if sensor.calibration_ok():
                log.success("BNO085 calibrated — stable state confirmed.")
                return True
            time.sleep(0.5)
        log.warning("BNO085 calibration timeout — proceeding anyway.")
        return False
    except HardwareError as exc:
        log.error(f"BNO085 FAIL: {exc}")
        return False


def stream(hz: int = 50) -> None:
    """Continuously print BNO085 readings. Press Ctrl+C to stop."""
    interval = 1.0 / hz
    log.info(f"BNO085 stream at {hz} Hz — Ctrl+C to stop.")
    try:
        sensor = BNO085()
        while True:
            print("\033[2J\033[H", end="")   # clear terminal
            data = sensor.read()
            print(_format(data))
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("BNO085 stream stopped.")
    except HardwareError as exc:
        log.error(f"BNO085 stream failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BNO085 hardware test")
    parser.add_argument("--stream", action="store_true", help="Run continuous stream")
    parser.add_argument("--cal",    action="store_true", help="Wait for calibration")
    parser.add_argument("--hz", type=int, default=50,   help="Stream rate (default 50)")
    args = parser.parse_args()

    if args.stream:
        stream(hz=args.hz)
    elif args.cal:
        calibration_wait()
    else:
        ok = quick_check()
        exit(0 if ok else 1)
