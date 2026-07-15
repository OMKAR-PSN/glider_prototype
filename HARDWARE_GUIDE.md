# HARDWARE_GUIDE.md
# CAN-7U-SAT Paraglider GNC — Hardware Reference

**This is a practical debugging reference, not a tutorial.**
Read PROJECT_STATE.md first to understand the overall architecture.
This document assumes you are holding the Raspberry Pi in your hand or staring at a
terminal connected to it and something is wrong.

---

## Section 1: File Map — "If X happens, go to file Y"

| Situation / Symptom | File to open | Specific function / class |
|---|---|---|
| Glider is not turning when it should | `flight_computer.py` | `FlightComputer.run()` — check `GUIDED_DESCENT` branch; verify `delta_a` is non-zero in log output |
| Glider turns the **wrong direction** | `flight_computer.py` -> `_obs_from_state()` | Check `heading_err` sign convention: `(target_bearing - aircraft_heading)`. Also verify servo wiring — left/right may be swapped physically |
| Deployment trigger fires at wrong altitude | `state_machine/flight_states.py` | `StateMachine.update()` — `DROGUE_DESCENT` branch checks `baro_history` average <= 600m AGL. Check `ground_altitude` passed to `StateMachine.__init__()` |
| Telemetry packets malformed or missing fields | `telemetry/packet.py` | `GliderPacket` dataclass — verify field count matches ground station parser. `telemetry/xbee_link.py` `XBeeLink.send()` for serial issues |
| GPS position not updating | `flight_computer.py` | `SITL_GPS.read()` or real hardware equivalent in `hw_interface/real_hardware.py`. Check `_last_gps_time` watchdog — if stale >500ms, RL is bypassed |
| IMU attitude angles look wrong (drifting or flipped) | `estimation/madgwick.py` | `MadgwickFilter.update()` — check axis sign conventions match physical mounting. `get_euler_angles()` returns `(roll, pitch, yaw)` in radians |
| RL model not loading | `flight_computer.py` | `_load_rl_model()` — check `config/gains.yaml` key `rl.onnx_model_path`. Startup log line `[RL] Model loaded` must appear. If missing, check `onnxruntime` install |
| RL model loads but produces bad servo commands | `flight_computer.py` | `_obs_from_state()` — add `print(obs)` to inspect all 16 values before inference. Cross-check against Section 5 expected ranges. See also `_validate_and_rescale()` |
| PID fallback engages unexpectedly | `flight_computer.py` | `run()` — look for `[FALLBACK]` in log output. Causes: NaN in obs, inference timeout >5ms, GPS stale >500ms, model not loaded |
| Watchdog fires / servos return to neutral | `flight_computer.py` | `_rl_inference()` for timeout watchdog; `run()` for GPS staleness check and loop overrun detection. Budget is 5ms inference, 50ms loop |
| Servo not responding / moving to wrong angle | `config/gains.yaml` | `servos.left_neutral_pwm` / `servos.right_neutral_pwm`. Check `hw_interface/real_hardware.py` for PCA9685 write calls. See Section 3 for calibration |
| Wind estimator producing wrong wind direction | `estimation/wind_estimator.py` | `WindEstimatorRLS.update()` and `get_wind_estimate()`. Needs 30-60s of circling to converge — see Section 6 |
| Altitude estimate wrong or jumpy | `estimation/ekf_altitude.py` | `EKFAltitude` — state `x[0,0]` is altitude, `x[1,0]` is vertical velocity. Check `R_baro=2.0` and `Q` values. BMP388 noise on real hardware is 0.5-1.0m RMS |
| State machine stuck in wrong state | `state_machine/flight_states.py` | `StateMachine.update()` — print `self.state`, `self.baro_history`, `vertical_velocity`. Common cause: EKF vertical_velocity not converging fast enough after drogue deploy |
| Camera not capturing | `sensors/drivers.py` | Camera stub not yet implemented — see PROJECT_STATE.md Task H |

---

## Section 2: Sensor Wiring Reference

### ICM-20948 (9-axis IMU — Gyro + Accel + Mag)
| Property | Value |
|---|---|
| Interface | I2C |
| I2C Address | `0x68` (AD0 low) or `0x69` (AD0 high) |
| RPi Bus | I2C-1 — SDA=GPIO2 (pin 3), SCL=GPIO3 (pin 5) |
| Python library | `icm20948` or `smbus2` raw |
| Driver location | `sensors/drivers.py` -> `ICM20948Driver` (stub) / `hw_interface/real_hardware.py` |
| Live test | `python -c "import smbus2; b=smbus2.SMBus(1); print(hex(b.read_byte_data(0x68,0x00)))"` -> should return `0xEA` (WHO_AM_I) |
| Wrong value symptom | Returns `0x00` or `0xFF` -> I2C not connected or address wrong |

### BMP388 (Barometric Pressure — Altitude)
| Property | Value |
|---|---|
| Interface | I2C |
| I2C Address | `0x76` (SDO low) or `0x77` (SDO high) |
| RPi Bus | I2C-1 — same SDA/SCL as IMU |
| Python library | `bmp3xx` or `adafruit-circuitpython-bmp3xx` |
| Driver location | `sensors/drivers.py` -> `BMP388Driver` (stub) |
| Live test | `python -c "import board, adafruit_bmp3xx; b=adafruit_bmp3xx.BMP3XX_I2C(board.I2C()); print(f'{b.altitude:.1f}m')"` |
| Wrong value symptom | Returns `nan` or `9999` -> sensor not found. Returns constant value -> I2C hang, power-cycle sensor |

### GPS (u-blox M8N or equivalent)
| Property | Value |
|---|---|
| Interface | UART |
| UART Port | `/dev/ttyAMA0` (RPi 4 primary UART, disable BT in `/boot/config.txt`) |
| Baud rate | `9600` (factory default) or `38400` if pre-configured |
| Python library | `pyserial` + `pynmea2` |
| Driver location | `hw_interface/real_hardware.py` -> `RealHardware.read_gps()` (stub to implement) |
| Live test | `python -c "import serial; s=serial.Serial('/dev/ttyAMA0',9600,timeout=1); print(s.readline())"` -> should see `$GPGGA,...` NMEA sentences |
| Wrong value symptom | Empty bytes -> wrong port or baud. `,,,,` in NMEA fields -> no satellite lock yet (wait 60-90s outdoors) |

### INA219 (Current/Power Monitor)
| Property | Value |
|---|---|
| Interface | I2C |
| I2C Address | `0x40` (A0=A1=GND) |
| RPi Bus | I2C-1 |
| Python library | `adafruit-circuitpython-ina219` |
| Driver location | `sensors/drivers.py` -> `INA219Driver` (stub) |
| Live test | `python -c "import board,adafruit_ina219; i=adafruit_ina219.INA219(board.I2C()); print(f'{i.bus_voltage:.2f}V {i.current:.1f}mA')"` |
| Wrong value symptom | Current reads `0.0mA` with load present -> shunt resistor not in circuit |

### PCA9685 (16-Channel PWM — Servo Controller)
| Property | Value |
|---|---|
| Interface | I2C |
| I2C Address | `0x40` (default — CONFLICTS with INA219 if both default; solder A0 jumper on one) |
| RPi Bus | I2C-1 |
| Python library | `adafruit-circuitpython-servokit` or `RPi.GPIO` + `smbus2` |
| Driver location | `hw_interface/real_hardware.py` -> `RealHardware.write_servos()` (stub to implement) |
| Live test | `python -c "from adafruit_servokit import ServoKit; kit=ServoKit(channels=16); kit.servo[0].angle=90"` -> left servo should move to neutral |
| Wrong value symptom | Servo buzzes but does not move -> PWM frequency wrong (set to 50Hz). Servo moves to extreme -> neutral offset incorrect (see Section 3) |

---

## Section 3: Servo Calibration Procedure

The MG996R servos are specified as 90 degrees each side (180 degrees total travel) at
standard 1000-2000us PWM. In practice, individual servos vary +/-5-10 degrees from nominal.

### Step 1 — Find true neutral (with glider assembled and brake lines connected)

```bash
python -c "
from adafruit_servokit import ServoKit
kit = ServoKit(channels=16)
kit.servo[0].angle = 90  # left servo
kit.servo[1].angle = 90  # right servo
print('Servos at 90 degrees. Observe brake line tension.')
input('Press Enter to exit.')
"
```

With the glider laid flat and brake lines attached:
- Both brake lines should have equal, light tension (not slack, not pulled)
- The trailing edge should be symmetric — hold a ruler across it
- If one side is pulled down, the trim is off for that servo

Adjust in 1-degree steps until tension is equal. The neutral angle that produces
equal tension is your true neutral.

### Step 2 — Verify 60-120 degree physical range

```bash
python -c "
from adafruit_servokit import ServoKit
import time
kit = ServoKit(channels=16)
kit.servo[0].angle = 60   # max brake pull
time.sleep(1)
kit.servo[0].angle = 120  # full release
time.sleep(1)
kit.servo[0].angle = 90   # neutral
print('60=max pull, 120=full release, 90=neutral')
"
```

### Step 3 — Update config/gains.yaml

The exact key to change:

```yaml
servos:
  left_neutral_pwm: 90      # <- change to your measured true neutral
  right_neutral_pwm: 90     # <- change independently for right servo
  min_angle: 60
  max_angle: 120
```

IMPORTANT: The active flight computer also hardcodes 90.0 in the servo mixing
expressions in flight_computer.py -> run():

    left_servo  = 90.0 + delta_s - delta_a
    right_servo = 90.0 + delta_s + delta_a

If your true neutral is not 90 degrees, update these constants to match.
This is the single most likely source of asymmetric flight on first hardware test.

---

## Section 4: First Power-On Checklist

Execute in order from cold hardware to running guidance loop.

    [ ] 1. Verify power supply: 5V 3A minimum to RPi. Measure voltage at GPIO
           with multimeter — should be 4.9-5.1V under load.

    [ ] 2. Verify I2C is enabled:
           sudo raspi-config -> Interface Options -> I2C -> Enable
           Reboot if just enabled.

    [ ] 3. Scan I2C bus to confirm sensors present:
           sudo i2cdetect -y 1
           Expected: 0x68 (ICM-20948), 0x76 (BMP388), 0x40 (PCA9685)

    [ ] 4. Test GPS serial:
           python -c "import serial; s=serial.Serial('/dev/ttyAMA0',9600,timeout=2); print(s.readline())"
           Should see $GPGGA or $GPRMC sentence. If empty: check port, baud, wiring.

    [ ] 5. Install Python dependencies (first time only):
           pip install onnxruntime stable-baselines3 numpy pyyaml pyserial

    [ ] 6. Verify ONNX model file:
           ls -lh models/sac_policy_6500000_16D.onnx
           Expected: ~9.5KB for .onnx, ~276KB for .onnx.data

    [ ] 7. Run SITL smoke test:
           cd /path/to/glider_gnc
           python flight_computer.py --sitl

    [ ] 8. Verify first lines of console output (see below).

    [ ] 9. When SITL output is correct, switch to real hardware mode
           (requires real hardware stubs in real_hardware.py to be implemented).

### Expected console output on successful startup:

    [HH:MM:SS] INFO Initializing Flight Computer...
    [HH:MM:SS] INFO --> Software-In-The-Loop (SITL) Mode ACTIVE
    [HH:MM:SS] INFO [RL] Model loaded | path=models/sac_policy_6500000_16D.onnx | sha256=ff74f871 | in_shape=[1, 16] | out_shape=[1, 2]
    [HH:MM:SS] INFO [RL] Shape check PASSED -- RL is PRIMARY controller.
    [HH:MM:SS] INFO Starting 20Hz Flight Loop...
    [HH:MM:SS] INFO [RL ] STATE:BOOST                  | ALT:   0.0m | DIST: 1000.0m | SRV L: 90.0 R: 90.0

If sha256 is not ff74f871: ONNX file replaced or corrupted.
If Shape check PASSED does not appear: obs/action dimension mismatch — do not fly.
If [RL] ONNX not found: check config/gains.yaml -> rl.onnx_model_path.
If PID-only mode: onnxruntime not installed or model file missing.

---

## Section 5: Observation Builder Quick Reference

All 16 dimensions as built by flight_computer.py -> _obs_from_state().
Must be identical to training/env.py -> _get_obs() or the policy produces garbage.

| dim | Name | Real sensor | Driver/function | Expected range | "Wrong" looks like |
|---|---|---|---|---|---|
| 0 | sin(heading_err) | ICM-20948 -> Madgwick yaw | att_filter.get_euler_angles()[2] | [-1, 1] | Constant 0.0 -> Madgwick not updating |
| 1 | cos(heading_err) | Same | Same | [-1, 1] | cos=1 always -> heading_err=0, always on track |
| 2 | dist / 1000.0 | GPS lat/lon | GPS position | [0, ~3] | Very large (>10) -> GPS not locked |
| 3 | alt_excess / 1000.0 | BMP388 + EKF | ekf_alt.altitude | [-1, 1] typical | Large negative -> way below glide slope |
| 4 | wind_speed / 10.0 | RLS wind estimator | wind_estimator.get_wind_estimate() | [0, 1] | 0.0 for first 30-60s -> normal, estimator converging |
| 5 | sin(wind_dir) | RLS wind estimator | Same | [-1, 1] | Oscillates for 30-60s then stabilises — normal |
| 6 | cos(wind_dir) | RLS wind estimator | Same | [-1, 1] | Same |
| 7 | pitch / 0.5 | ICM-20948 -> Madgwick | att_filter.get_euler_angles()[1] | [-2, 2] | Drifting without motion -> Madgwick beta too low |
| 8 | roll / 0.5 | ICM-20948 -> Madgwick | att_filter.get_euler_angles()[0] | [-2, 2] | Same |
| 9 | yaw_rate / 0.5 | ICM-20948 gyro z | raw gz from imu.read() | [-2, 2] | Constant non-zero -> IMU bias |
| 10 | prev_delta_a / 30.0 | Previous control output | fc.prev_delta_a | [-1, 1] | Constant 0.0 -> action not stored (first step OK) |
| 11 | prev_delta_s / 30.0 | Previous control output | fc.prev_delta_s | [0, 1] | Same |
| 12 | sin(track_err) | GPS course-over-ground | math.atan2(gvy, gvx) | [-1, 1] | Identical to obs[0] -> zero wind (OK) or wrong heading source |
| 13 | cos(track_err) | GPS course-over-ground | Same | [-1, 1] | Same |
| 14 | lateral_drift / 8.0 | GPS ground velocity | GPS speed + heading | [-2, 2] | Constant 0.0 -> GPS speed zero or not locked |
| 15 | time_to_impact / 200.0 | BMP388 alt + wind est. | altitude / sink_rate / 200, cap 2.0 | [0, 2] | 2.0 always -> very high or sink_rate near zero |

CRITICAL ANGLE CONVENTION (see flight_computer.py _obs_from_state docstring):
  obs[0-1]   heading_err uses aircraft body-frame YAW from Madgwick filter
  obs[12-13] track_err   uses GPS course-over-ground (COG)
  These diverge in crosswind. Using the wrong source for either signal breaks the policy.

---

## Section 6: Known Sim-to-Real Differences to Expect

### GPS update rate
- Simulation: 20 Hz (every loop tick)
- Real hardware: 1 Hz default; configurable to 5 Hz via UBX protocol
- Impact: At 1 Hz the glider travels ~12m between GPS fixes at cruise speed
- Fix: Configure GPS to 5 Hz before flight. Also set gps_staleness_timeout_ms to
  1100ms in config/gains.yaml if running at 1 Hz, or RL will be permanently bypassed.

### GPS heading vs Madgwick yaw
- gps_heading (COG): used for track_err, lateral_drift, and PID fallback heading compute
- yaw (Madgwick): used for heading_err only
- On real hardware: GPS COG is noisy below ~2 m/s groundspeed. Madgwick yaw is smooth
  but accumulates drift over time. Verify Madgwick yaw against compass reference pre-flight.

### Brake line mechanical lag
- Simulation: servo commands are instant (same loop tick)
- Real hardware: MG996R takes 0.1-0.2s for a full 60-degree move
- Impact: 2-4 frames of lag per command. The policy uses prev_delta_a/prev_delta_s
  (obs[10-11]) for implicit lag compensation. Expect slightly more overshoot on
  real hardware than in SITL — this is expected and acceptable.

### Wind estimator convergence time — MOST IMPORTANT FOR FIRST HARDWARE TEST
- Simulation: wind is sampled from WindModel directly; estimator converges in seconds
  because wind is constant and the policy environment has stable conditions.
- Real hardware: the RLS wind estimator starts COLD — wx=0, wy=0. It estimates wind
  by comparing GPS ground velocity to inertial velocity derived from heading + airspeed.
  On a real drop, it needs approximately 30-60 seconds of consistent flight across
  varying headings to converge to the actual wind vector.
- What this means for flight:
    - During the first 30-60 seconds of guided descent, obs[4-6] (wind speed and
      direction) will read near zero and be unreliable.
    - The policy was trained with randomised wind and is robust to uncertainty,
      but crosswind correction will be delayed while the estimator catches up.
    - obs[14] lateral_drift is computed directly from GPS velocity and is ALWAYS
      correct — this is the primary crosswind correction signal until the wind
      estimate converges. The policy will still fly, just with less precision.
    - After ~60s, if the glider is crabbing at a consistent offset angle, the
      wind estimate has converged. Before that, slightly erratic corrections are normal.
- Logging: add a log line for wind_estimator.get_wind_estimate() every 20 frames
  to watch convergence in real time during the first flight test.

### Altitude trigger reliability
- Simulation: BMP388 altitude is perfect (equals dynamics.altitude exactly)
- Real hardware: BMP388 has +/-0.5-1.0m RMS noise; sensitive to airflow over the port.
  Deployment shock can cause 10-20m transient spikes.
- Fix: Mount BMP388 in a ported enclosure (static port), not open to airstream.
  The 10-sample rolling average in the state machine filters single-sample spikes.

### Control loop timing jitter
- Simulation: exactly 20 Hz
- Real hardware: RPi 4 is not real-time. Expect +/-2-5ms jitter under normal load.
  Monitor [WATCHDOG] Loop overrun log lines. If frequent under thermal throttling,
  reduce inference load or increase inference_timeout_ms in config/gains.yaml.

---

## Section 7: Emergency Recovery During a Real Flight Test

In priority order if the system misbehaves mid-flight:

### 1. Manual RC override
CURRENT STATE: No hardware RC override is wired. This is a critical gap before
first real flight. Required action: wire a PWM override switch on the RC receiver
that physically disconnects PCA9685 PWM and substitutes RC receiver servo signals.

### 2. What PID fallback looks like in telemetry
When PID fallback engages, the log line prefix changes from [RL ] to [PID]:

    [HH:MM:SS] WARNING [FALLBACK] RL exception: Inference timeout: 6.2ms > 5ms -- engaging PID.
    [HH:MM:SS] INFO [PID] STATE:GUIDED_DESCENT | ALT: 350.0m | DIST: 800.0m | SRV L: 95.0 R: 85.0

PID fallback uses guidance/heading_pid.py to steer toward target.
Performance is degraded (no wind compensation) but glider will attempt to reach target.

### 3. If telemetry is lost (XBee link drops)
The flight computer continues running — telemetry loss does not affect guidance.
XBee failures are silently swallowed in telemetry/xbee_link.py -> XBeeLink.send().
On the ground: the glider is now fully autonomous. Do not run toward it. Wait for landing.
Recovery: walk to landing site with GPS to record miss distance for post-flight analysis.

### 4. Safe landing procedure if guidance fails completely
- Note wind direction and glider position
- The glider will continue descending — it cannot gain altitude
- At 5-10m AGL the glider slows significantly due to ground effect
- Do not attempt to catch from below — brake lines will tangle
- Post-flight: read flight_computer.log on the SD card to identify the failure state
  and obs values at the time of failure

---

*Last updated: 2026-07-11*
*All Tasks 1-9 integration tests PASS — see PROJECT_STATE.md Task G for full evidence record*
*Next step: Implement real RPi hardware drivers in hw_interface/real_hardware.py and sensors/drivers.py*
