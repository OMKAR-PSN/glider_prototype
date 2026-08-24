"""BMP388 SPI barometer hardware test — adapted from TARSR/GARUD raspi_test_code.

Usage (on Pi):
    python -m tests.test_bmp388_hw             # quick one-shot check
    python -m tests.test_bmp388_hw --stream    # continuous stream at 50 Hz
"""

from __future__ import annotations

import argparse
import time

from sensors.drivers import BMP388, BaroData
from utils.helpers import HardwareError
from utils.logger import build_logger

log = build_logger(name="test_bmp388", save_logs=True)


def _format(d: BaroData) -> str:
    return (
        f"Temperature: {d.temperature:+.2f} °C\n"
        f"Pressure:    {d.pressure:.2f} hPa\n"
        f"Altitude:    {d.altitude:.2f} m AGL"
    )


def quick_check() -> bool:
    """Read one BMP388 sample — PASS/FAIL."""
    log.info("BMP388 quick check starting...")
    try:
        sensor = BMP388()
        data = sensor.read()
        log.success(f"BMP388 OK  |  {data.temperature:.1f}°C  "
                    f"|  {data.pressure:.1f} hPa  "
                    f"|  AGL {data.altitude:.1f} m")
        return True
    except HardwareError as exc:
        log.error(f"BMP388 FAIL: {exc}")
        return False


def stream(hz: int = 50) -> None:
    """Continuously print BMP388 readings. Press Ctrl+C to stop."""
    interval = 1.0 / hz
    log.info(f"BMP388 stream at {hz} Hz — Ctrl+C to stop.")
    try:
        sensor = BMP388()
        while True:
            print("\033[2J\033[H", end="")   # clear terminal
            data = sensor.read()
            print(_format(data))
            time.sleep(interval)
    except KeyboardInterrupt:
        log.info("BMP388 stream stopped.")
    except HardwareError as exc:
        log.error(f"BMP388 stream failed: {exc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BMP388 hardware test")
    parser.add_argument("--stream", action="store_true", help="Run continuous stream")
    parser.add_argument("--hz", type=int, default=50,   help="Stream rate (default 50)")
    args = parser.parse_args()

    if args.stream:
        stream(hz=args.hz)
    else:
        ok = quick_check()
        exit(0 if ok else 1)
