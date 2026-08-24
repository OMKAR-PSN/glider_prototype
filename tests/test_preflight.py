"""Pre-flight hardware validation — run ALL checks before every drop.

Adapted from TARSR/GARUD raspi_test_code/tests/test_all.py.
Extended to cover GARUD GNC sensors: BNO085, BMP388, INA219, PCA9685, ONNX model.

Usage (on Pi, from glider_gnc root):
    python -m tests.test_preflight

Exit code 0 = ALL PASS, exit code 1 = at least one FAIL.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from utils.helpers import (
    HardwareError,
    get_system_info,
    list_spi_devices,
    scan_i2c_bus,
)
from utils.logger import build_logger

log = build_logger(name="preflight", save_logs=True)

# Expected I2C addresses on the GARUD HAT
EXPECTED_I2C = {0x40: "PCA9685 (servo driver)", 0x41: "INA219 (power monitor)"}

# ONNX model path (relative to glider_gnc root)
ONNX_MODEL = Path("models/sac_policy_6500000_16D.onnx")


# ---------------------------------------------------------------------------
# Individual checks — each returns True (PASS) or False (FAIL)
# ---------------------------------------------------------------------------

def check_system_info() -> bool:
    """Print system diagnostics header."""
    try:
        info = get_system_info()
        log.info(f"Host       : {info.hostname}")
        log.info(f"OS         : {info.os_version}")
        log.info(f"Python     : {info.python_version}")
        log.info(f"CPU Temp   : {info.cpu_temperature_c} °C")
        log.info(f"RAM        : {info.ram_usage}")
        log.info(f"Disk       : {info.disk_usage}")
        log.info(f"IP         : {info.ip_address}")
        return True
    except Exception as exc:
        log.warning(f"System info unavailable: {exc}")
        return True   # non-critical, don't fail preflight for this


def check_spi_bus() -> bool:
    """Verify SPI bus is enabled and devices are present."""
    devices = list_spi_devices()
    if devices:
        log.success(f"SPI devices: {', '.join(str(d) for d in devices)}")
        return True
    log.error("No SPI devices found. Enable SPI with: sudo raspi-config → Interface Options → SPI")
    return False


def check_i2c_bus() -> bool:
    """Scan I2C bus and verify expected devices are present."""
    try:
        found = scan_i2c_bus()
        found_hex = [hex(a) for a in found]
        log.info(f"I2C scan found: {found_hex}")
        all_ok = True
        for addr, name in EXPECTED_I2C.items():
            if addr in found:
                log.success(f"  {hex(addr)} {name} — FOUND")
            else:
                log.error(f"  {hex(addr)} {name} — MISSING")
                all_ok = False
        return all_ok
    except HardwareError as exc:
        log.error(f"I2C scan failed: {exc}")
        return False


def check_bno085() -> bool:
    """Initialise BNO085 and read one sample."""
    try:
        from sensors.drivers import BNO085
        sensor = BNO085()
        data = sensor.read()
        log.success(f"BNO085 — Cal={data.calibration_status}  "
                    f"Yaw={data.yaw:+.3f} rad  "
                    f"Accel_z={data.accel_z:+.2f} m/s²")
        return True
    except HardwareError as exc:
        log.error(f"BNO085 FAIL: {exc}")
        return False


def check_bmp388() -> bool:
    """Initialise BMP388 and read one sample."""
    try:
        from sensors.drivers import BMP388
        sensor = BMP388()
        data = sensor.read()
        log.success(f"BMP388 — {data.temperature:.1f}°C  "
                    f"{data.pressure:.1f} hPa  "
                    f"AGL {data.altitude:.1f} m")
        return True
    except HardwareError as exc:
        log.error(f"BMP388 FAIL: {exc}")
        return False


def check_ina219() -> bool:
    """Initialise INA219 and read voltage/current."""
    try:
        from sensors.drivers import INA219
        sensor = INA219(address=0x41)
        data = sensor.read()
        if data is None:
            log.error("INA219 returned None")
            return False
        log.success(f"INA219 — {data.voltage:.2f} V  {data.current * 1000:.1f} mA")
        return True
    except Exception as exc:
        log.error(f"INA219 FAIL: {exc}")
        return False


def check_pca9685() -> bool:
    """Initialise PCA9685 and set all channels to neutral (90°)."""
    try:
        from hw_interface.real_hardware import RealHardware
        hw = RealHardware()
        hw.write_servos(90.0, 90.0)   # left=neutral, right=neutral
        log.success("PCA9685 — servos set to neutral (90°)")
        return True
    except Exception as exc:
        log.error(f"PCA9685 FAIL: {exc}")
        return False


def check_onnx_model() -> bool:
    """Verify ONNX model file exists, loads, and produces a valid output."""
    if not ONNX_MODEL.exists():
        log.error(f"ONNX model NOT FOUND: {ONNX_MODEL}")
        return False
    try:
        import numpy as np
        import onnxruntime as ort
        session = ort.InferenceSession(str(ONNX_MODEL))
        dummy_obs = np.zeros((1, 16), dtype=np.float32)
        t0 = time.monotonic()
        output = session.run(None, {session.get_inputs()[0].name: dummy_obs})
        elapsed_ms = (time.monotonic() - t0) * 1000
        raw = output[0][0]
        if not all(map(lambda x: -10 < x < 10, raw)):
            log.error(f"ONNX output out of expected range: {raw}")
            return False
        log.success(f"ONNX model — loaded OK  |  inference {elapsed_ms:.1f} ms  |  output {raw}")
        return True
    except Exception as exc:
        log.error(f"ONNX model FAIL: {exc}")
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_all() -> bool:
    log.info("=" * 55)
    log.info("  GARUD PRE-FLIGHT HARDWARE VALIDATION")
    log.info("=" * 55)

    check_system_info()

    results: dict[str, bool] = {
        "SPI Bus"   : check_spi_bus(),
        "I2C Bus"   : check_i2c_bus(),
        "BNO085"    : check_bno085(),
        "BMP388"    : check_bmp388(),
        "INA219"    : check_ina219(),
        "PCA9685"   : check_pca9685(),
        "ONNX Model": check_onnx_model(),
    }

    print()
    print("=" * 40)
    print("  PRE-FLIGHT SUMMARY")
    print("=" * 40)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<14} {status}")
    overall = all(results.values())
    print("-" * 40)
    print(f"  {'Overall':<14} {'GO FOR LAUNCH' if overall else 'NO-GO — FIX FAILURES'}")
    print("=" * 40)

    return overall


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
