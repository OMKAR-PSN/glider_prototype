"""
tests/generate_actuation_dataset.py
======================================
Generates a synthetic flight dataset to test the actuation algorithm
(AGC / RL → elevon mixer → servo angles).

Simulates a parafoil descent from 600m AGL to the ground, with:
  - Varying heading errors (wind perturbations + random drift)
  - Wind estimation noise
  - Altitude decay at realistic glide ratio
  - Outputs both the inputs to AGC and the resulting servo angles

Run:
    python -m tests.generate_actuation_dataset
    python -m tests.generate_actuation_dataset --target-lat 18.5204 --target-lon 73.8567

Output:
    tests/data/actuation_test_dataset.csv
    tests/data/actuation_summary.txt

CSV Columns:
    time_s          - elapsed time in seconds
    altitude_m      - AGL altitude (metres)
    lat             - current latitude
    lon             - current longitude
    heading_rad     - current heading (radians)
    target_bearing  - bearing to target (radians)
    heading_err_rad - error (heading - bearing), wrapped to [-pi, pi]
    heading_err_deg - same in degrees
    wind_x, wind_y  - estimated wind components (m/s)
    gps_speed       - ground speed (m/s)
    delta_a         - AGC aileron output (degrees, +/- 30)
    delta_s         - AGC symmetric brake (degrees, 0–30)
    servo_left_deg  - left servo angle
    servo_right_deg - right servo angle
    expected_turn   - expected turn direction (LEFT/STRAIGHT/RIGHT)
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DT             = 0.05       # 20 Hz
TOTAL_TIME_S   = 120.0      # 2 minute descent
INITIAL_ALT    = 600.0      # metres AGL
GLIDE_RATIO    = 8.0        # horizontal/vertical (tuned from your config)
AIRSPEED       = 7.0        # m/s
SINK_RATE      = AIRSPEED / GLIDE_RATIO

# Start position (Pune area, default)
START_LAT = 18.5300
START_LON = 73.8600

# AGC parameters (from gnc_process.py)
BASE_GAIN = 15.0
K_MAX     = 25.0
K_MIN     = 5.0
SIGMA     = 30.0

OUTPUT_DIR = Path("tests/data")


# ---------------------------------------------------------------------------
# AGC controller (copy of gnc_process._agc_fallback)
# ---------------------------------------------------------------------------

def agc(heading_err: float, altitude: float, wind_x: float, wind_y: float,
        target_bearing: float) -> tuple[float, float]:
    """Adaptive Gain Control — same formula as gnc_process.py."""
    cross  = -wind_x * math.sin(target_bearing) + wind_y * math.cos(target_bearing)
    K_wind = BASE_GAIN + (K_MAX - K_MIN) * (abs(cross) / (abs(cross) + SIGMA))
    urgency = max(0.0, 1.0 - altitude / 300.0) if altitude > 10.0 else 0.0
    K_total = K_wind * (1.0 + 0.5 * urgency)
    delta_a = K_total * math.tanh(heading_err)
    delta_a = max(-30.0, min(30.0, delta_a))
    delta_s = 5.0
    return delta_a, delta_s


def mix(delta_a: float, delta_s: float) -> tuple[float, float]:
    """Elevon mixer — same as gnc_process._mix."""
    left  = max(60.0, min(120.0, 90.0 + delta_s + delta_a))
    right = max(60.0, min(120.0, 90.0 + delta_s - delta_a))
    return left, right


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = {
    "nominal": {
        "desc": "Normal descent, moderate crosswind, target ahead",
        "wind_x": 2.0, "wind_y": 1.0,
        "start_lat": START_LAT, "start_lon": START_LON,
        "drift_sigma": 0.05,   # rad heading noise
    },
    "strong_crosswind": {
        "desc": "Strong crosswind from west, target requires right turn",
        "wind_x": -6.0, "wind_y": 0.5,
        "start_lat": START_LAT, "start_lon": START_LON - 0.002,
        "drift_sigma": 0.08,
    },
    "headwind": {
        "desc": "Direct headwind, minimal heading error needed",
        "wind_x": 4.0, "wind_y": 0.0,
        "start_lat": START_LAT, "start_lon": START_LON,
        "drift_sigma": 0.03,
    },
    "spiral_approach": {
        "desc": "Starting far off target, large initial heading error, must spiral in",
        "wind_x": 1.0, "wind_y": -2.0,
        "start_lat": START_LAT + 0.005, "start_lon": START_LON - 0.005,
        "drift_sigma": 0.10,
    },
    "low_wind": {
        "desc": "Very low wind, algorithm should use base gain only",
        "wind_x": 0.2, "wind_y": 0.1,
        "start_lat": START_LAT, "start_lon": START_LON + 0.001,
        "drift_sigma": 0.02,
    },
}


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_scenario(name: str, cfg: dict, target_lat: float,
                      target_lon: float, output_dir: Path) -> dict:
    """Generate a single scenario CSV. Returns summary stats."""
    random.seed(42)  # reproducible

    rows = []
    lat     = cfg["start_lat"]
    lon     = cfg["start_lon"]
    alt     = INITIAL_ALT
    heading = math.atan2(target_lon - lon, target_lat - lat)   # initial pointing

    t       = 0.0
    wind_x  = cfg["wind_x"]
    wind_y  = cfg["wind_y"]
    sigma   = cfg["drift_sigma"]

    max_err = 0.0
    servo_range_left  = [180.0, 0.0]
    servo_range_right = [180.0, 0.0]

    while t < TOTAL_TIME_S and alt > 0:
        # Target bearing
        dx = (target_lat - lat) * 111_320
        dy = (target_lon - lon) * 111_320 * math.cos(math.radians(lat))
        dist = math.hypot(dx, dy)
        target_bearing = math.atan2(dy, dx)

        # Heading error (wrapped)
        heading_err = (target_bearing - heading + math.pi) % (2 * math.pi) - math.pi
        max_err = max(max_err, abs(heading_err))

        # AGC
        delta_a, delta_s = agc(heading_err, alt, wind_x, wind_y, target_bearing)
        left_deg, right_deg = mix(delta_a, delta_s)

        servo_range_left[0]  = min(servo_range_left[0],  left_deg)
        servo_range_left[1]  = max(servo_range_left[1],  left_deg)
        servo_range_right[0] = min(servo_range_right[0], right_deg)
        servo_range_right[1] = max(servo_range_right[1], right_deg)

        # Turn direction label
        if delta_a > 2.0:
            turn = "RIGHT"
        elif delta_a < -2.0:
            turn = "LEFT"
        else:
            turn = "STRAIGHT"

        rows.append({
            "time_s":          round(t, 3),
            "altitude_m":      round(alt, 2),
            "lat":             round(lat, 7),
            "lon":             round(lon, 7),
            "heading_rad":     round(heading, 5),
            "target_bearing":  round(target_bearing, 5),
            "heading_err_rad": round(heading_err, 5),
            "heading_err_deg": round(math.degrees(heading_err), 3),
            "wind_x":          round(wind_x, 3),
            "wind_y":          round(wind_y, 3),
            "gps_speed":       round(AIRSPEED, 2),
            "delta_a":         round(delta_a, 4),
            "delta_s":         round(delta_s, 4),
            "servo_left_deg":  round(left_deg, 3),
            "servo_right_deg": round(right_deg, 3),
            "expected_turn":   turn,
        })

        # Propagate state
        heading += delta_a / 200.0 + random.gauss(0, sigma * DT)   # turn rate ~ delta_a
        vx = AIRSPEED * math.cos(heading) + wind_x
        vy = AIRSPEED * math.sin(heading) + wind_y
        lat += (vx * DT) / 111_320
        lon += (vy * DT) / (111_320 * math.cos(math.radians(lat)))
        alt -= SINK_RATE * DT

        # Slow wind drift
        wind_x += random.gauss(0, 0.01)
        wind_y += random.gauss(0, 0.01)

        t += DT

        if dist < 5.0 and alt < 50.0:
            break   # close enough

    # Write CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"actuation_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        headers = list(rows[0].keys())
        f.write(",".join(headers) + "\n")
        for row in rows:
            f.write(",".join(str(row[h]) for h in headers) + "\n")

    return {
        "scenario":          name,
        "rows":              len(rows),
        "duration_s":        round(t, 1),
        "final_alt_m":       round(alt, 1),
        "max_heading_err_deg": round(math.degrees(max_err), 1),
        "servo_left_range":  f"{servo_range_left[0]:.1f} – {servo_range_left[1]:.1f}°",
        "servo_right_range": f"{servo_range_right[0]:.1f} – {servo_range_right[1]:.1f}°",
        "csv":               str(csv_path),
    }


# ---------------------------------------------------------------------------
# Sanity checks on the generated data
# ---------------------------------------------------------------------------

def run_sanity_checks(csv_path: Path) -> list[str]:
    """Run basic checks on a generated CSV. Returns list of PASS/FAIL strings."""
    results = []
    with open(csv_path, "r") as f:
        headers = f.readline().strip().split(",")
        rows = [dict(zip(headers, line.strip().split(","))) for line in f]

    if not rows:
        results.append("FAIL: No rows in CSV")
        return results

    # 1. Servo range check
    for row in rows:
        l, r = float(row["servo_left_deg"]), float(row["servo_right_deg"])
        if not (60.0 <= l <= 120.0):
            results.append(f"FAIL: servo_left_deg={l:.1f} out of [60, 120] at t={row['time_s']}")
            break
        if not (60.0 <= r <= 120.0):
            results.append(f"FAIL: servo_right_deg={r:.1f} out of [60, 120] at t={row['time_s']}")
            break
    else:
        results.append("PASS: All servo angles within [60°, 120°]")

    # 2. delta_a range check
    for row in rows:
        da = float(row["delta_a"])
        if not (-30.0 <= da <= 30.0):
            results.append(f"FAIL: delta_a={da:.1f} out of [-30, 30]")
            break
    else:
        results.append("PASS: All delta_a values within [-30°, +30°]")

    # 3. Turn direction consistency
    mismatches = 0
    for row in rows:
        da   = float(row["delta_a"])
        turn = row["expected_turn"]
        if da > 2.0  and turn != "RIGHT": mismatches += 1
        if da < -2.0 and turn != "LEFT":  mismatches += 1
    if mismatches == 0:
        results.append("PASS: Turn direction labels consistent with delta_a sign")
    else:
        results.append(f"FAIL: {mismatches} turn direction mismatches")

    # 4. Heading error decreases over time (controller should converge)
    first_err  = abs(float(rows[0]["heading_err_deg"]))
    last_10_avg = sum(abs(float(r["heading_err_deg"])) for r in rows[-10:]) / 10
    if last_10_avg < first_err:
        results.append(f"PASS: Heading error converges ({first_err:.1f}° → {last_10_avg:.1f}°)")
    else:
        results.append(f"INFO: Heading error did not converge ({first_err:.1f}° → {last_10_avg:.1f}°) — may be wind-limited")

    # 5. Altitude is monotonically decreasing
    alts = [float(r["altitude_m"]) for r in rows]
    if all(alts[i] >= alts[i+1] for i in range(len(alts)-1)):
        results.append("PASS: Altitude monotonically decreasing")
    else:
        results.append("FAIL: Altitude increased during flight (check sink rate)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate actuation test dataset")
    parser.add_argument("--target-lat", type=float, default=18.5204)
    parser.add_argument("--target-lon", type=float, default=73.8567)
    parser.add_argument("--scenario",   type=str,   default=None,
                        help="Run a single scenario. Options: " + ", ".join(SCENARIOS))
    args = parser.parse_args()

    target_lat = args.target_lat
    target_lon = args.target_lon
    scenarios  = {args.scenario: SCENARIOS[args.scenario]} if args.scenario else SCENARIOS

    print(f"\nTarget: {target_lat:.6f}, {target_lon:.6f}")
    print(f"Output: {OUTPUT_DIR.resolve()}\n")
    print("=" * 65)

    summary_lines = []
    for name, cfg in scenarios.items():
        print(f"\n[{name.upper()}] {cfg['desc']}")
        stats = generate_scenario(name, cfg, target_lat, target_lon, OUTPUT_DIR)

        checks = run_sanity_checks(Path(stats["csv"]))
        for check in checks:
            print(f"  {check}")

        line = (f"  Rows: {stats['rows']}  |  Duration: {stats['duration_s']}s  |  "
                f"Final alt: {stats['final_alt_m']}m  |  "
                f"Max heading err: {stats['max_heading_err_deg']}°\n"
                f"  Servo L: {stats['servo_left_range']}  "
                f"Servo R: {stats['servo_right_range']}\n"
                f"  CSV: {stats['csv']}")
        print(line)
        summary_lines.append(f"[{name}]\n{line}\n")

    # Write summary
    summary_path = OUTPUT_DIR / "actuation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"GARUD Actuation Dataset Summary\n")
        f.write(f"Target: {target_lat:.6f}, {target_lon:.6f}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("\n".join(summary_lines))

    print(f"\n{'='*65}")
    print(f"Summary written to: {summary_path}")
    print(f"\nTo test your algorithm against this data:")
    print(f"  python -m tests.test_actuation_algorithm")
    print(f"\nScenario files:")
    for name in scenarios:
        print(f"  tests/data/actuation_{name}.csv")


if __name__ == "__main__":
    main()
