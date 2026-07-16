import time
import math
import sys
import os
import hashlib
import logging

# Ensure the module can be run from root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from estimation.madgwick import MadgwickFilter
from estimation.ekf_altitude import EKFAltitude
from guidance.heading_pid import HeadingPID
from state_machine.flight_states import StateMachine, FlightState
from sim.dynamics import GliderDynamics
from sim.wind_model import WindModel
from hw_interface.simulated_hardware import SimulatedHardware
from hw_interface.real_hardware import RealHardware
from estimation.wind_estimator import WindEstimatorRLS
import yaml
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
log = logging.getLogger("FlightComputer")
log.setLevel(logging.INFO)
fh = logging.FileHandler("flight_computer.log")
fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S'))
# We specifically do NOT add a StreamHandler to the root logger here.
# The FlightDashboard will handle console rendering. If rich is missing,
# __main__ will attach a standard StreamHandler.
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(fh)

# ---------------------------------------------------------------------------
# Hardware Interface Stubs
# ---------------------------------------------------------------------------
class SITL_IMU:
    def __init__(self, hw):
        self.hw = hw
    def read(self):
        imu = self.hw.read_imu()
        return imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_p, imu.gyro_q, imu.gyro_r, imu.mag_x, imu.mag_y, imu.mag_z

class SITL_Baro:
    def __init__(self, hw):
        self.hw = hw
    def read_altitude(self):
        return self.hw.read_baro().altitude

class SITL_GPS:
    """
    BUG FIX (2026-07-09): Previously hardcoded lat=18.52, lon=73.85 (Pune coordinates).
    This caused curr_x = 1,852,000m and curr_y = 7,385,000m in the flight loop.
    Now correctly passes through the simulated x/y via the 1e-5 lat/lon convention.
    """
    def __init__(self, hw):
        self.hw = hw
    def read(self):
        gps = self.hw.read_gps()
        return gps.latitude, gps.longitude, gps.altitude, gps.ground_speed, gps.heading

class SITL_Servos:
    def __init__(self, hw):
        self.hw = hw
    def write(self, left_pwm, right_pwm):
        self.hw.write_servos(left_pwm, right_pwm)

class HW_IMU:
    def __init__(self, hw):
        self.hw = hw
    def read(self):
        imu = self.hw.read_imu()
        return imu.accel_x, imu.accel_y, imu.accel_z, imu.gyro_p, imu.gyro_q, imu.gyro_r, imu.mag_x, imu.mag_y, imu.mag_z

class HW_Baro:
    def __init__(self, hw):
        self.hw = hw
    def read_altitude(self):
        return self.hw.read_baro().altitude

class HW_GPS:
    def __init__(self, hw):
        self.hw = hw
    def read(self):
        gps = self.hw.read_gps()
        return gps.latitude, gps.longitude, gps.altitude, gps.ground_speed, gps.heading

class HW_Servos:
    def __init__(self, hw):
        self.hw = hw
    def write(self, left_pwm, right_pwm):
        self.hw.write_servos(left_pwm, right_pwm)

class DummyTelemetry:
    def send(self, packet):
        pass


# ---------------------------------------------------------------------------
# FlightComputer
# ---------------------------------------------------------------------------
class FlightComputer:
    # Physical bounds for post-rescaling output validation
    DELTA_A_MIN = -30.0
    DELTA_A_MAX =  30.0
    DELTA_S_MIN =   0.0
    DELTA_S_MAX =  30.0

    def __init__(self, use_simulator=True):
        log.info("Initializing Flight Computer...")

        self.dt = 0.05  # 20 Hz loop
        self.use_simulator = use_simulator

        if self.use_simulator:
            log.info("--> Software-In-The-Loop (SITL) Mode ACTIVE")
            self.sim_dynamics = GliderDynamics(-1000.0, -1000.0, 0.0, math.radians(45))
            self.sim_wind = WindModel(2.0, math.radians(90))
            self.sim_hw = SimulatedHardware(self.sim_dynamics)

            self.imu    = SITL_IMU(self.sim_hw)
            self.baro   = SITL_Baro(self.sim_hw)
            self.gps    = SITL_GPS(self.sim_hw)
            self.servos = SITL_Servos(self.sim_hw)
        else:
            log.info("--> REAL FLIGHT Mode ACTIVE")
            self.real_hw = RealHardware()
            self.real_hw.initialize()
            self.imu    = HW_IMU(self.real_hw)
            self.baro   = HW_Baro(self.real_hw)
            self.gps    = HW_GPS(self.real_hw)
            self.servos = HW_Servos(self.real_hw)

        self.telemetry = DummyTelemetry()

        self.target_x = 0.0
        self.target_y = 0.0

        self.att_filter     = MadgwickFilter(beta=0.1)
        self.ekf_alt        = EKFAltitude(self.dt, initial_alt=self.baro.read_altitude())
        self.heading_pid    = HeadingPID(kp=10.0, ki=0.1, kd=1.0, output_limit=30.0)
        self.state_machine  = StateMachine(ground_altitude=0.0)
        self.wind_estimator = WindEstimatorRLS()
        self.prev_delta_a   = 0.0
        self.prev_delta_s   = 0.0

        self._last_gps_time = time.time()

        with open("config/gains.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.glide_ratio         = self.config['airframe']['glide_ratio']
        self.gps_timeout_s       = self.config['telemetry']['gps_staleness_timeout_ms'] / 1000.0
        self.inference_timeout_s = self.config['rl']['inference_timeout_ms'] / 1000.0

        self.rl_session = None
        self.rl_active  = False
        self._load_rl_model()

    def _sha256_short(self, path, chars=8):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
        return h.hexdigest()[:chars]

    def _load_rl_model(self):
        if ort is None:
            log.warning("[RL] onnxruntime not installed -- PID-only mode.")
            return

        model_path    = self.config['rl']['onnx_model_path']
        expected_obs  = self.config['rl']['obs_dim']
        expected_act  = self.config['rl']['action_dim']

        if not os.path.exists(model_path):
            log.warning(f"[RL] ONNX not found at '{model_path}' -- PID-only mode.")
            return

        try:
            session = ort.InferenceSession(model_path)
            inp = session.get_inputs()[0]
            out = session.get_outputs()[0]
            sha = self._sha256_short(model_path)
            log.info(
                f"[RL] Model loaded | path={model_path} | sha256={sha} | "
                f"in_shape={inp.shape} | out_shape={out.shape}"
            )
            if inp.shape != [1, expected_obs]:
                raise ValueError(f"Input shape mismatch: expected [1,{expected_obs}], got {inp.shape}")
            if out.shape != [1, expected_act]:
                raise ValueError(f"Output shape mismatch: expected [1,{expected_act}], got {out.shape}")

            self.rl_session    = session
            self.rl_input_name = inp.name
            self.rl_active     = True
            log.info("[RL] Shape check PASSED -- RL is PRIMARY controller.")
        except Exception as e:
            log.error(f"[RL] Load failed: {e} -- falling back to PID.")
            self.rl_session = None
            self.rl_active  = False

    def _obs_from_state(self, curr_x, curr_y, target_bearing, dist,
                        alt_excess, pitch, roll, yaw_rate,
                        gps_speed, gps_heading, altitude,
                        aircraft_heading):
        """
        16D observation builder. Must match training/env.py _get_obs() exactly.

        Angle convention (verified against env.py):
          aircraft_heading : body-frame yaw (where the nose points) — from attitude filter.
                             Used for heading_err (obs[0-1]) only.
          gps_heading      : course-over-ground (where the glider is actually moving,
                             including wind drift) — from GPS ground track.
                             Used for track_err (obs[12-13]) and lateral_drift (obs[14]).
          These are equal in zero-wind; they diverge in crosswind conditions, which is
          exactly the information obs[0-1] vs obs[12-13] encodes for the agent.

        obs[15] time_to_impact is capped at 2.0 (preserved from env.py line 116).
        """
        # obs[0-1]: heading_err — gap between nose direction and target bearing
        heading_err = (target_bearing - aircraft_heading + math.pi) % (2 * math.pi) - math.pi

        wx, wy = self.wind_estimator.get_wind_estimate()
        wind_speed = math.hypot(wx, wy)
        wind_dir   = math.atan2(wy, wx)

        gvx = gps_speed * math.cos(gps_heading)
        gvy = gps_speed * math.sin(gps_heading)

        course_over_ground = math.atan2(gvy, gvx)
        track_err = (target_bearing - course_over_ground + math.pi) % (2 * math.pi) - math.pi

        lateral_drift      = -gvx * math.sin(target_bearing) + gvy * math.cos(target_bearing)
        lateral_drift_norm = lateral_drift / 8.0

        airspeed_approx    = math.hypot(gvx - wx, gvy - wy)
        sink_rate          = max(airspeed_approx / self.glide_ratio, 0.1)
        time_to_impact_norm = min(altitude / sink_rate / 200.0, 2.0)

        return np.array([[
            math.sin(heading_err),     # obs[0]
            math.cos(heading_err),     # obs[1]
            dist / 1000.0,             # obs[2]
            alt_excess / 1000.0,       # obs[3]
            wind_speed / 10.0,         # obs[4]
            math.sin(wind_dir),        # obs[5]
            math.cos(wind_dir),        # obs[6]
            pitch / 0.5,               # obs[7]
            roll / 0.5,                # obs[8]
            yaw_rate / 0.5,            # obs[9]
            self.prev_delta_a / 30.0,  # obs[10]
            self.prev_delta_s / 30.0,  # obs[11]
            math.sin(track_err),       # obs[12]
            math.cos(track_err),       # obs[13]
            lateral_drift_norm,        # obs[14]
            time_to_impact_norm,       # obs[15]
        ]], dtype=np.float32)

    def _validate_and_rescale(self, raw):
        """
        Rescales raw tanh outputs to physical units and validates.
        Raises ValueError on NaN, Inf, or out-of-range values.
        """
        r0, r1 = float(raw[0]), float(raw[1])
        if not math.isfinite(r0) or not math.isfinite(r1):
            raise ValueError(f"NaN/Inf in ONNX output: [{r0}, {r1}]")
        delta_a = r0 * 30.0
        delta_s = (r1 + 1.0) / 2.0 * 30.0
        if not (self.DELTA_A_MIN <= delta_a <= self.DELTA_A_MAX):
            raise ValueError(f"delta_a={delta_a:.2f} out of range")
        if not (self.DELTA_S_MIN <= delta_s <= self.DELTA_S_MAX):
            raise ValueError(f"delta_s={delta_s:.2f} out of range")
        return delta_a, delta_s

    def _rl_inference(self, obs):
        """Runs ONNX inference with watchdog. Raises RuntimeError on timeout."""
        t0  = time.perf_counter()
        raw = self.rl_session.run(None, {self.rl_input_name: obs})[0][0]
        elapsed = time.perf_counter() - t0
        if elapsed > self.inference_timeout_s:
            raise RuntimeError(f"Inference timeout: {elapsed*1000:.1f}ms > {self.inference_timeout_s*1000:.0f}ms")
        return raw

    def run(self, dashboard=None):
        log.info("Starting 20Hz Flight Loop...")
        frame_id = 0

        while True:
            loop_start = time.time()

            if self.use_simulator:
                wx, wy = self.sim_wind.get_wind()
                self.sim_dynamics.step(self.dt, wx, wy)
                if self.sim_dynamics.altitude <= 0:
                    log.info("--> Simulation Finished. Glider has landed.")
                    break

            # 1. Read Sensors
            ax, ay, az, gx, gy, gz, mx, my, mz = self.imu.read()
            baro_alt = self.baro.read_altitude()
            lat, lon, gps_alt, gps_speed, gps_heading = self.gps.read()

            curr_x = lat / 1e-5
            curr_y = lon / 1e-5

            now = time.time()
            gps_fresh = (now - self._last_gps_time) <= self.gps_timeout_s
            if not gps_fresh:
                log.warning(f"[WATCHDOG] GPS stale >{self.gps_timeout_s*1000:.0f}ms -- fallback to PID.")
            else:
                self._last_gps_time = now

            v_gx = gps_speed * math.cos(gps_heading)
            v_gy = gps_speed * math.sin(gps_heading)
            self.wind_estimator.update(v_gx, v_gy, gps_heading)

            # 2. State Estimation
            self.att_filter.update(ax, ay, az, gx, gy, gz, mx, my, mz, self.dt)
            roll, pitch, yaw = self.att_filter.get_euler_angles()

            accel_z_earth_down = (-math.sin(pitch) * ax
                                  + math.sin(roll) * math.cos(pitch) * ay
                                  + math.cos(roll) * math.cos(pitch) * az)
            self.ekf_alt.predict(9.81 - accel_z_earth_down)
            self.ekf_alt.update_baro(baro_alt)

            # 3. State Machine
            state = self.state_machine.update(self.ekf_alt.altitude, self.ekf_alt.vertical_velocity)

            # 4. Guidance
            left_servo  = 90.0
            right_servo = 90.0
            controller_used = "NEUTRAL"
            rl_succeeded = False
            delta_a = 0.0
            delta_s = 0.0
            obs = None

            if state == FlightState.GUIDED_DESCENT:
                aim_x = self.target_x
                aim_y = self.target_y
                target_bearing = math.atan2(aim_y - curr_y, aim_x - curr_x)
                dist       = math.hypot(aim_x - curr_x, aim_y - curr_y)
                alt_excess = self.ekf_alt.altitude - (dist / self.glide_ratio)

                if self.rl_active and gps_fresh:
                    try:
                        obs = self._obs_from_state(
                            curr_x, curr_y, target_bearing, dist, alt_excess,
                            pitch, roll, gz,
                            gps_speed, gps_heading, self.ekf_alt.altitude,
                            aircraft_heading=yaw
                        )
                        raw     = self._rl_inference(obs)
                        delta_a, delta_s = self._validate_and_rescale(raw)
                        rl_succeeded    = True
                        controller_used = "RL"
                    except Exception as e:
                        log.warning(f"[FALLBACK] RL exception: {e} -- engaging PID.")

                if not rl_succeeded:
                    gains = self.config['gain_schedules']
                    if self.ekf_alt.altitude > gains['cruise']['min_alt_agl']:
                        self.heading_pid.kp = gains['cruise']['heading_kp']
                        self.heading_pid.ki = gains['cruise']['heading_ki']
                        self.heading_pid.kd = gains['cruise']['heading_kd']
                    elif self.ekf_alt.altitude > gains['approach']['min_alt_agl']:
                        self.heading_pid.kp = gains['approach']['heading_kp']
                        self.heading_pid.ki = gains['approach']['heading_ki']
                        self.heading_pid.kd = gains['approach']['heading_kd']
                    else:
                        self.heading_pid.kp = gains['final']['heading_kp']
                        self.heading_pid.ki = gains['final']['heading_ki']
                        self.heading_pid.kd = gains['final']['heading_kd']
                    delta_a = self.heading_pid.compute(target_bearing, gps_heading, self.dt)
                    delta_s = 30.0 if self.ekf_alt.altitude < 10.0 else 0.0
                    controller_used = "PID"

                self.prev_delta_a = delta_a
                self.prev_delta_s = delta_s

                left_servo  = max(60.0, min(120.0, 90.0 + delta_s - delta_a))
                right_servo = max(60.0, min(120.0, 90.0 + delta_s + delta_a))

            self.servos.write(left_servo, right_servo)

            # 5. Telemetry
            packet = (f"{frame_id},{loop_start:.2f},{lat},{lon},{gps_alt},{baro_alt},"
                      f"{math.degrees(roll):.1f},{math.degrees(pitch):.1f},{math.degrees(yaw):.1f}")
            self.telemetry.send(packet)

            if dashboard:
                wx, wy = self.wind_estimator.get_wind_estimate()
                dashboard.update(
                    state_name=state.name,
                    controller=controller_used,
                    baro_alt=baro_alt,
                    dist=math.hypot(curr_x - self.target_x, curr_y - self.target_y),
                    roll=roll, pitch=pitch, yaw=yaw,
                    gps_speed=gps_speed, gps_heading=gps_heading,
                    wind_speed=math.hypot(wx, wy),
                    wind_dir=math.atan2(wy, wx),
                    left_servo=left_servo, right_servo=right_servo,
                    delta_a=delta_a if state == FlightState.GUIDED_DESCENT else 0.0,
                    delta_s=delta_s if state == FlightState.GUIDED_DESCENT else 0.0,
                    obs=obs if rl_succeeded else None
                )

            # 6. Loop timing enforcement
            elapsed = time.time() - loop_start
            if elapsed > self.dt:
                log.warning(f"[WATCHDOG] Loop overrun: {elapsed*1000:.1f}ms > {self.dt*1000:.0f}ms budget")
            else:
                time.sleep(self.dt - elapsed)

            frame_id += 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CAN-7U-SAT Flight Computer")
    parser.add_argument("--sitl", action="store_true", help="Run in SITL mode")
    args = parser.parse_args()
    
    from telemetry.dashboard import FlightDashboard
    dashboard = FlightDashboard()
    
    if not dashboard.enabled:
        # Fallback to standard console logging if rich is not installed
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(ch)
        
    fc = FlightComputer(use_simulator=args.sitl)
    
    dashboard.start()
    try:
        fc.run(dashboard=dashboard)
    except KeyboardInterrupt:
        log.info("Flight Computer shutdown safely.")
    finally:
        dashboard.stop()
