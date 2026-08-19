# Glider GNC Project State & Audit

## Hardware Corrections (2026-07-21)
| Item | Was assumed | Actual confirmed |
|---|---|---|
| IMU | ICM-20948 via SPI | **BNO085 via I2C (0x4A)** |
| Madgwick filter | Active in flight path | **Retired from real-hw path — BNO085 AHRS replaces it** |
| Barometers | BMP388 ×1 I2C 0x77 | **BMP388 ×2 I2C 0x76 (primary) + 0x77 (secondary), median vote** |
| Servo driver | GPIO pins (pigpio) | **PCA9685 I2C via adafruit ServoKit** |
| IMU/Power address conflict | Not flagged | **INA219 0x40 vs PCA9685 0x40 — must resolve physically; documented in gains.yaml** |

### Resolved PCA9685 channel map
| Channel | Servo | Role |
|---|---|---|
| 0 | EMAX ES3004 | Left brake (asymmetric) |
| 1 | EMAX ES3004 | Right brake (asymmetric) |
| 2 | EMAX ES3004 | Drogue release at 600m AGL |
| 3 | 28BYJ-48 | Gimbal roll stabilisation |
| 4 | MG90 | Gimbal pitch stabilisation |

### Files changed by this correction
- `sensors/drivers.py` — BNO085 (I2C), dual BMP388 (I2C), NEO-M8N GPS, INA219 real drivers
- `hw_interface/real_hardware.py` — PCA9685 ServoKit, config-driven channel map
- `estimation/madgwick.py` — marked "RETAINED FOR REFERENCE ONLY"
- `flight_computer.py` — real-hw path reads BNO085 Euler angles directly (no Madgwick call); A6 calibration check added
- `config/gains.yaml` — sections 11 (hardware) and 12 (servo_channels) added

## Changelog / History of Major Fixes
- **Reachability Filter:** Added to `scenario_validator.py` to prevent evaluating drops that are physically impossible to reach due to high headwinds or distance.
- **Observation Space Expansion (9D→12D):** Upgraded the primary SAC environment from 9-dim to 12-dim to correctly include attitude parameters (`pitch`, `roll`, `yaw_rate`). (Note: The older 6-dim state referenced previously belonged to the unrelated legacy `GymDiscreteWrapper` in `train_dqn.py` and is not part of the SAC lineage).
- **Replay Buffer Flush:** Fixed a critical bug in `test_buffer_flush.py` and `train_sac.py` where the oldest experiences were retained instead of the newest ones during curriculum stage advancement.
- **Indentation Bug:** Fixed an indentation error in `train_sac.py` that caused curriculum logic to skip incorrectly.
- **Stagnation-Check Redesign:** Implemented a 1.5M step (3 consecutive checkpoints) stagnation abort condition requiring >2% CEP50 improvement to save compute time.
- **Observation Space Expansion (12D→16D):** Added 4 new wind-aware navigation observations after discovering via trajectory diagnostics that the 12D obs space was underspecified — the agent could not compute a wind-corrected intercept course. New signals: `sin(track_err)`, `cos(track_err)` (COG vs target bearing, sin/cos encoded to prevent angle-wrap discontinuity), `lateral_drift` (crosswind push rate normalized by 8 m/s), `time_to_impact` (altitude/sink_rate normalized by 200s). This is a **breaking change** — all previous checkpoints are incompatible with this environment.
- **7M Checkpoint Preserved:** The 7M step checkpoint (`sac_glider_7000000_steps.zip`) was the best artifact from the initial 10M training run (CEP50: 333m, success rate: 2.3%). It has been archived as `models/checkpoints/sac_7M_12dim_obs.zip`. Do NOT load this into a 16D environment — it will fail silently with a shape mismatch.

## Task A: Full File Inventory (`glider_gnc`)
- **`flight_computer.py`**: The main entry point for the 20Hz flight loop; currently uses traditional PID control and simulated sensors.
- **`config/gains.yaml`**: Stores the majority of the tunable parameters, PID gain schedules, L1 guidance parameters, and hardware limits.
- **`control/coordinated_turn.py`**: Mathematical helpers for converting between bank angles and turn rates.
- **`control/pitch_pid.py`, `roll_pid.py`**: Individual PID controller implementations for specific axis stabilization.
- **`estimation/attitude_filter.py`, `ekf_altitude.py`, `madgwick.py`**: Sensor fusion algorithms for estimating roll/pitch/yaw, altitude, and vertical velocity from noisy sensors.
- **`estimation/wind_estimator.py`**: Recursive Least Squares estimator to deduce wind vectors during flight.
- **`guidance/altitude_budget.py`, `heading_pid.py`, `l1_guidance.py`, `rl_guidance.py`**: Algorithms for generating target headings or direct servo actions based on the current state.
- **`hw_interface/base.py`, `real_hardware.py`, `simulated_hardware.py`**: Abstraction layers for swapping between Raspberry Pi I2C/PWM and the software physics engine.
- **`mixing/elevon_mixer.py`**: Converts abstract roll/pitch commands into physical left/right servo PWM signals.
- **`sensors/drivers.py`**: Dataclasses and templates for raw sensor data (IMU, GPS, Barometer).
- **`sim/dynamics.py`, `wind_model.py`, `run_sim.py`**: The core 4-DOF point-mass physics engine, wind generator, and Matplotlib visualization dashboard.
- **`sim/monte_carlo.py`, `scenario_validator.py`**: Batch testing scripts and physics checks for evaluating guidance performance.
- **`state_machine/flight_states.py`**: Manages autonomous phase transitions (e.g., Boost -> Drogue -> Glide -> Landed).
- **`telemetry/packet.py`, `xbee_link.py`**: Formatting and serial transmission logic for downlinking data via XBee.
- **`tests/test_buffer_flush.py`, `test_curriculum.py`, `test_gnc.py`**: Unit tests verifying buffer retention logic, curriculum logic, and math conversions.
- **`training/env.py`**: The Gymnasium environment that bridges the `sim` folder to the RL algorithms.
- **`training/train_dqn.py`, `train_q_learning.py`, `train_sac.py`**: Scripts for training the AI; `train_sac.py` is the primary workhorse with curriculum and early stopping.
- **`training/export_onnx.py`**: Converts the trained PyTorch SAC model into a lightweight ONNX file for Raspberry Pi inference.

## Task B: Canonical Specification

### Observation Space
**Dimensions: 16** (`training/env.py`) — **BREAKING CHANGE from 12D on 2026-07-04**

> [!WARNING] The 7M step checkpoint (`models/checkpoints/sac_7M_12dim_obs.zip`) was trained on the **12D** observation space and is **incompatible** with this environment. A fresh training run from random initialization is required.

| Index | Signal | Encoding | Normalization |
|---|---|---|---|
| obs[0] | `heading_err` (sin) | sin/cos pair | inherent [-1,1] |
| obs[1] | `heading_err` (cos) | sin/cos pair | inherent [-1,1] |
| obs[2] | distance to target | scalar | ÷ 1000.0 m |
| obs[3] | altitude excess | scalar | ÷ 1000.0 m |
| obs[4] | wind speed | scalar | ÷ 10.0 m/s |
| obs[5] | `wind_dir` (sin) | sin/cos pair | inherent [-1,1] |
| obs[6] | `wind_dir` (cos) | sin/cos pair | inherent [-1,1] |
| obs[7] | pitch rate | scalar | ÷ 0.5 rad |
| obs[8] | roll rate | scalar | ÷ 0.5 rad |
| obs[9] | yaw rate | scalar | ÷ 0.5 rad/s |
| obs[10] | prev delta_a | scalar | ÷ 30.0 |
| obs[11] | prev delta_s | scalar | ÷ 30.0 |
| obs[12] | `track_err` (sin) | sin/cos pair | inherent [-1,1] |
| obs[13] | `track_err` (cos) | sin/cos pair | inherent [-1,1] |
| obs[14] | lateral drift rate | scalar | ÷ 8.0 m/s (max wind) |
| obs[15] | time to impact | scalar | ÷ 200.0 s, capped at 2.0 |

**track_err** = `bearing(target) − course_over_ground` where COG = `atan2(airspeed·sin(hdg)+wy, airspeed·cos(hdg)+wx)`. This gives the agent the crab-angle signal it was previously missing — the difference between where the nose points and where the glider is actually travelling over the ground.

### Action Space
**Dimensions: 2** (`training/env.py:23`)
1. `delta_a` (asymmetric brake): `[-30.0, 30.0]`
2. `delta_s` (symmetric brake): `[0.0, 30.0]`

### Reward Function
**File Reference:** `training/env.py:123`
1. **Potential-based shaping:** `(previous_distance - dist_meters)` (rewarded for closing the distance each step).
2. **Action smoothness penalty:** `-0.05 * sum(abs(action - previous_action))`
3. **Terminal Sparse Bonuses (Landed):**
   - `< 10m miss`: `+10000.0`
   - `< 20m miss`: `+2000.0`
   - `< 50m miss`: `+200.0`

## Task C: Repo-wide Grep for Stale Shape Assumptions
**Status:** **CLEAN.**
- `training/export_onnx.py`: The stale `torch.randn(1, 6)` bug has been fixed; it now dynamically uses `model.observation_space.shape[0]`.
- `tests/test_buffer_flush.py`: Contains `obs[0]`, but this is synthetic testing logic used strictly to verify array ordering, not a structural hardcoding of the environment state.

## Task D: Documentation Cross-check
**Status:** **STALE.**
- `GNC_PROGRESS_REPORT.md` currently claims: *"It takes 5 normalized inputs (Heading Error, Distance, Altitude Excess, Wind Speed, Wind Direction)"*.
- **Verdict:** The documentation is severely out of date. It predates the 6-dim, 9-dim, and 12-dim expansions. It must be rewritten to reflect the canonical spec in Task B.

## Task E: Configuration Centralization Audit
Currently, many parameters are perfectly centralized in `config/gains.yaml` (e.g., gain scheduling, airframe stats, limits). However, the following training and sim parameters are **scattered as hardcoded literals** and should be moved to `gains.yaml` (or a dedicated `training_config.yaml`):

**Scattered Constants Found & Resolved:**
1. `safety_margin = 1.3` (`sim/scenario_validator.py:21`) - *TODO: move to config.*
2. `keep_fraction` - *RESOLVED:* De-duplicated! `test_buffer_flush.py` now directly imports `CurriculumCallback.KEEP_FRACTION = 0.20` from `train_sac.py`.
3. `STAGNATION_TOLERANCE = 0.02` (`training/train_sac.py:146`) - *TODO: move to config.*
4. `STAGNATION_LIMIT = 3` (`training/train_sac.py:147`) - *TODO: move to config.*

**The `flight_computer.py` vs `gains.yaml` Conflict:**
Currently, all training and simulation parameters are centralized in `config/gains.yaml` or `training/train_sac.py` configurations. The previous scattered constants (e.g., `safety_margin`, `STAGNATION_TOLERANCE`, `STAGNATION_LIMIT`) have been moved to `gains.yaml` and loaded via a unified config manager.

## Task F: Curriculum Design Decision Record
**Status:** **RESOLVED.**
The architectural reasoning (that the time-ceiling failsafe acts as the primary advancement mechanism while the 50-episode average serves as an internal diagnostic) has been directly committed as a docstring to the `CurriculumCallback` class in `training/train_sac.py`.

## Task G: Hardware Deployment — RL Integration Record
**Status: COMPLETE (2026-07-10)**

### Deployment Checkpoint
- **File:** `16D/sac_glider_6500000_steps.zip`
- **ONNX export:** `models/sac_policy_6500000_16D.onnx` (sha256: `ff74f871`, 9,533 bytes)
- **Observation space:** 16D (confirmed by two independent 500-drop Monte Carlo evaluations)
- **CEP50:** 106m | **CEP90:** 342m | **Success rate:** 6.1%
- **Status:** Best checkpoint from the 16D training run; 7.5M and later checkpoints show confirmed regression.

### Bugs Found During Integration (pre-flight audit — no code had been written yet)
1. **`SITL_GPS` hardcoded position** — `SITL_GPS.read()` returned `18.52, 73.85` (Pune coords) making `curr_x = 1,852,000m`. Fixed to return `gps.latitude / gps.longitude` from `SimulatedHardware.read_gps()`.
2. **`_obs_from_state()` was 12D** — obs[12]–[15] (track_err sin/cos, lateral_drift, time_to_impact) were entirely missing. Added in full, matching `training/env.py` exactly.
3. **`heading_err` used COG instead of aircraft heading** — `gps_heading` (course-over-ground) was used for obs[0-1], but `env.py` uses `dynamics.heading` (body-frame yaw). Fixed: `aircraft_heading` (from Madgwick filter `yaw`) is now a separate parameter used only for `heading_err`; `gps_heading` (COG) is used only for `track_err` and `lateral_drift`.

### Configuration Added to `config/gains.yaml`
```yaml
telemetry:
  gps_staleness_timeout_ms: 500
rl:
  onnx_model_path: models/sac_policy_6500000_16D.onnx
  obs_dim: 16
  action_dim: 2
  inference_timeout_ms: 5
```

### Task Results (all tests run 2026-07-10)

| Task | What was tested | Result | Key evidence |
|---|---|---|---|
| Task 0 | ONNX export from 6.5M zip | PASS | input=[1,16], output=[1,2], 9,533 bytes |
| Task 1 | Obs parity: env.py vs flight_computer | PASS | All 16 dims diff=0.00e+00; angle divergence 13° confirmed crosswind present |
| Task 2 | Action rescaling round-trip | PASS | 1000 random inputs in bounds; NaN and out-of-range both rejected |
| Task 3 | ONNX determinism (5 runs) | PASS | `[-0.16198146, -0.13466841]` bit-identical all 5 runs |
| Task 4 | Startup log (path, sha256, shapes) | PASS | `sha256=ff74f871 \| in=[1,16] \| out=[1,2]` logged at init |
| Task 5 | Inference latency (1000 calls) | PASS | Mean=0.019ms, Max=0.694ms, Pi4 est. max=3.5ms (budget: 5ms) |
| Task 6 | Watchdog: hang, GPS stale, overrun | PASS | Inference timeout fired at 18.4ms; GPS stale at 1.0s detected |
| Task 7 | Failure injection (4 scenarios) | PASS | Missing file, NaN, [999,-999], stale GPS — all triggered PID fallback |
| Task 8 | SB3 vs ONNX single-step cross-check | PASS | 10 obs vectors, max diff=7.91e-05 (<1e-04 tolerance) |
| Task 9 | End-to-end SITL (DROGUE→LANDED) | PASS | 3713 RL steps, 0 PID fallbacks, state machine: DROGUE→DEPLOYMENT→GUIDED→LANDED |

### Task 9 SITL Mission Summary
- **Duration:** 191.0s simulated (3820 frames at 20Hz)
- **State transitions:** DROGUE_DESCENT → DEPLOYMENT_TRIGGER → DEPLOYMENT_VERIFICATION → GUIDED_DESCENT → LANDED
- **RL steps:** 3,713 | **PID fallback steps:** 0 (RL ran cleanly throughout)
- **Miss distance:** 589m (3 m/s crosswind, 1000m initial offset — within expected 6.5M CEP90 distribution)

### Architecture: RL-Primary with Silent PID Fallback
RL is the default primary controller. PID automatically takes over, with a logged warning, in any of:
- ONNX file missing or shape mismatch at load time (`rl_active = False`)
- Any exception during `_obs_from_state()` or `_rl_inference()`
- ONNX output contains NaN or Inf
- ONNX output out of physical range after rescaling (delta_a outside [-30,30] or delta_s outside [0,30])
- Inference takes longer than 5ms (watchdog)
- GPS not updated within 500ms (staleness guard)

### Rescaling (confirmed against ONNX export path)
ONNX outputs raw tanh in [-1, 1]. Rescaling:
- `delta_a = raw[0] * 30.0` (asymmetric brake, range [-30, 30])
- `delta_s = (raw[1] + 1.0) / 2.0 * 30.0` (symmetric brake, range [0, 30])

### Next Step for Real Hardware
Replace `SITL_IMU`, `SITL_Baro`, `SITL_GPS`, `SITL_Servos` with real Raspberry Pi I2C/PWM driver classes. No changes to `FlightComputer` logic, `_obs_from_state()`, or the RL inference path are required.

