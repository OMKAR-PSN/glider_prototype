import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.animation as animation
import numpy as np
import yaml
import time
import math
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.dynamics import GliderDynamics
from sim.wind_model import WindModel
from hw_interface.simulated_hardware import SimulatedHardware
from guidance.heading_pid import HeadingPID
from guidance.altitude_budget import AltitudeBudget
from guidance.rl_guidance import RLGuidance
from estimation.attitude_filter import ComplementaryFilter
from estimation.madgwick import MadgwickFilter
from estimation.ekf_altitude import EKFAltitude
from estimation.wind_estimator import WindEstimatorRLS
from state_machine.flight_states import StateMachine, FlightState

def load_config(path="config/gains.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Warning: Could not find {path}, using defaults.")
        # Fallback minimal config
        return {
            "roll_pid": {"kp": 1.5, "ki": 0.1, "kd": 0.05, "integral_limit": 10.0},
            "pitch_pid": {"kp": 0.8, "ki": 0.05, "kd": 0.02, "integral_limit": 5.0, "target_pitch_deg": -3.0},
            "l1_guidance": {"period": 20.0, "damping": 0.7, "max_bank_angle_deg": 35.0},
            "airframe": {"glide_ratio": 5.0, "nominal_airspeed": 15.0, "turn_rate_constant": 1.0},
            "servos": {"min_pwm_deg": 60.0, "max_pwm_deg": 120.0, "neutral_deg": 90.0, "left_elevon_trim": 0.0, "right_elevon_trim": 0.0},
            "estimation": {"attitude_alpha": 0.98, "wind_rls_lambda": 0.98},
            "altitude_budget": {"s_turn_excess_threshold": 15.0, "s_turn_bank_angle_deg": 20.0},
            "timing": {"target_hz": 20}
        }

class SimRunner:
    def __init__(self):
        self.config = load_config()
        self.reset_sim()

        # Target landing position
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_alt = 0.0

    def reset_sim(self):
        c = self.config
        self.dt = 1.0 / c["timing"]["target_hz"]
        self.sim_time = 0.0

        # Initial conditions: 1000m away, 0m altitude (Ground)
        self.dynamics = GliderDynamics(-1000.0, -1000.0, 0.0, math.radians(45))
        self.wind = WindModel(2.0, math.radians(90)) # 2m/s East wind
        self.hw = SimulatedHardware(self.dynamics)
        
        c = self.config
        
        # Parafoil PID (tuned for ~30 deg brake max)
        self.heading_pid = HeadingPID(kp=10.0, ki=0.1, kd=1.0, output_limit=30.0)
        
        self.alt_budget = AltitudeBudget(c["airframe"]["glide_ratio"], c["altitude_budget"]["s_turn_excess_threshold"], math.radians(c["altitude_budget"]["s_turn_bank_angle_deg"]))
        
        self.att_filter = MadgwickFilter(beta=0.1)
        self.ekf_alt = EKFAltitude(self.dt, initial_alt=0.0)
        self.wind_estimator = WindEstimatorRLS(c["estimation"]["wind_rls_lambda"])
        self.state_machine = StateMachine(ground_altitude=0.0)
        
        # RL Guidance (optional)
        self.rl_guidance = RLGuidance("models/sac_model.zip", "sac")

        self.history = {"x": [], "y": [], "alt": [], "heading_err": [], "state": []}

    def step(self):
        wx, wy = self.wind.get_wind()
        self.dynamics.step(self.dt, wx, wy)
        self.sim_time += self.dt

        imu = self.hw.read_imu()
        baro = self.hw.read_baro()
        gps = self.hw.read_gps()

        # Estimation
        # Madgwick expects rad/s, m/s^2, and Mag (we'll pass dummy mag for sim)
        self.att_filter.update(imu.accel_x, imu.accel_y, imu.accel_z, 
                               imu.gyro_p, imu.gyro_q, imu.gyro_r, 
                               imu.mag_x, imu.mag_y, imu.mag_z, self.dt) # use simulated mag
        roll_est, pitch_est, yaw_est = self.att_filter.get_euler_angles()
        
        # Calculate Earth frame Z acceleration for EKF
        # Body to Earth Z = -sin(theta)*ax + sin(phi)*cos(theta)*ay + cos(phi)*cos(theta)*az
        accel_z_earth_down = -math.sin(pitch_est) * imu.accel_x + math.sin(roll_est) * math.cos(pitch_est) * imu.accel_y + math.cos(roll_est) * math.cos(pitch_est) * imu.accel_z
        accel_z_earth_up = 9.81 - accel_z_earth_down
        
        # Update EKF with baro and GPS
        self.ekf_alt.predict(accel_z_earth_up)
        self.ekf_alt.update_baro(baro.altitude)
        self.ekf_alt.update_gps(gps.altitude)
        
        # Pass GPS ground velocity components and instantaneous compass heading!
        v_ground_x = gps.ground_speed * math.cos(gps.heading)
        v_ground_y = gps.ground_speed * math.sin(gps.heading)
        compass_heading = math.atan2(imu.mag_y, imu.mag_x)
        est_wx, est_wy, est_va = self.wind_estimator.update(v_ground_x, v_ground_y, compass_heading)

        # Use fused altitude for state machine
        fused_altitude = self.ekf_alt.altitude
        state = self.state_machine.update(fused_altitude, self.ekf_alt.vertical_velocity)

        # Guidance and Control
        if state == FlightState.GUIDED_DESCENT:
            # Wind Compensation
            nominal_airspeed = self.config["airframe"]["nominal_airspeed"]
            # Estimate time to target using actual EKF vertical velocity
            # vertical_velocity is positive UP, so sink rate is negative of that.
            # Use a moving average or just bound it to avoid division by zero.
            actual_sink_rate = -self.ekf_alt.vertical_velocity
            # Bound sink rate to reasonable values (e.g., 1.0 to 10.0 m/s) to prevent aim point from shooting to infinity
            sink_rate = max(1.0, min(10.0, actual_sink_rate))
            
            time_to_target = max(0.0, fused_altitude) / sink_rate
            
            aim_x = self.target_x - est_wx * time_to_target
            aim_y = self.target_y - est_wy * time_to_target
            
            # Guidance: Aim at wind-compensated target
            target_heading = math.atan2(aim_y - self.dynamics.y, aim_x - self.dynamics.x)
            
            # Gain Scheduling based on altitude
            if fused_altitude > 200.0:
                self.heading_pid.kp = 5.0
                self.heading_pid.ki = 0.05
                self.heading_pid.kd = 0.5
            elif fused_altitude > 50.0:
                self.heading_pid.kp = 10.0
                self.heading_pid.ki = 0.1
                self.heading_pid.kd = 1.0
            else:
                self.heading_pid.kp = 20.0
                self.heading_pid.ki = 0.2
                self.heading_pid.kd = 2.0
            
            # Control: Direct Heading PID outputs asymmetric brake (delta_a)
            delta_a = self.heading_pid.compute(target_heading, gps.heading, self.dt)
            
            # Constant sine wave wobble to excite the wind estimator!
            # Fade out the wobble below 100m to ensure precise final tracking
            wobble_gain = 5.0 if fused_altitude > 100.0 else (5.0 * (max(0.0, fused_altitude) / 100.0))
            wobble = wobble_gain * math.sin(self.sim_time * 2.0 * math.pi / 5.0) 
            delta_a += wobble
            
            # Symmetric brake (delta_s) for flare or descent rate control
            # We will use 0 deg during cruise, and apply 30 deg when below 10m to flare.
            delta_s = 30.0 if fused_altitude < 10.0 else 0.0
            
            # Parafoil Servo Mixing
            left_servo = 90.0 + delta_s - delta_a
            right_servo = 90.0 + delta_s + delta_a
            
            # Clamp to physical linkage limits [60, 120]
            left_servo = max(60.0, min(120.0, left_servo))
            right_servo = max(60.0, min(120.0, right_servo))
            
            self.hw.write_servos(left_servo, right_servo)
        
        elif state == FlightState.DROGUE_DESCENT:
            self.hw.trigger_drogue()

        # Logging
        target_bearing = math.atan2(self.target_y - self.dynamics.y, self.target_x - self.dynamics.x)
        heading_err = (target_bearing - self.dynamics.heading + math.pi) % (2*math.pi) - math.pi
        
        self.history["x"].append(self.dynamics.x)
        self.history["y"].append(self.dynamics.y)
        self.history["alt"].append(self.dynamics.altitude)
        self.history["heading_err"].append(math.degrees(heading_err))
        self.history["state"].append(state.name)

        return self.dynamics.altitude > 0

def create_dashboard():
    runner = SimRunner()
    
    # Apply Narada Theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 9), facecolor='#0D1117')
    fig.canvas.manager.set_window_title('Narada Flight Simulation Dashboard')
    
    # 3D Flight Path (See the glider)
    ax_map = fig.add_subplot(221, projection='3d')
    ax_map.set_facecolor('#0D1117')
    
    # Altitude Plot
    ax_alt = fig.add_subplot(222)
    ax_alt.set_facecolor('#111111')
    
    # Heading Error Plot
    ax_err = fig.add_subplot(223)
    ax_err.set_facecolor('#111111')
    
    # Info Panel
    ax_info = fig.add_subplot(224)
    ax_info.set_facecolor('#080A0C')
    ax_info.axis('off')

    plt.subplots_adjust(bottom=0.25)
    
    axcolor = '#1E1E1E'
    ax_wind_spd = plt.axes([0.2, 0.1, 0.65, 0.03], facecolor=axcolor)
    slider_wind_spd = Slider(ax_wind_spd, 'Wind Speed', 0.0, 10.0, valinit=2.0, color='#00E5FF')
    slider_wind_spd.label.set_color('#E8F6FF')
    
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'RESET SIM', color='#FFA000', hovercolor='#FFB300')
    btn_reset.label.set_color('#212121')
    btn_reset.label.set_weight('bold')

    def update_wind(val):
        runner.wind.speed = slider_wind_spd.val

    slider_wind_spd.on_changed(update_wind)

    # 3D Glider Path
    line_map, = ax_map.plot([], [], [], color='#00E5FF', linewidth=2, label='Glider Path')
    target_point, = ax_map.plot([runner.target_x], [runner.target_y], [runner.target_alt], marker='*', color='#FFA000', markersize=12, label='Target (Launch Pad)')
    ax_map.set_title("3D Flight Path", color='#00E5FF', fontname='Segoe UI', fontsize=14)
    ax_map.xaxis.pane.fill = False
    ax_map.yaxis.pane.fill = False
    ax_map.zaxis.pane.fill = False
    ax_map.grid(color='#37474F', linestyle='--', linewidth=0.5)
    ax_map.legend(facecolor='#1E1E1E', edgecolor='#555555', labelcolor='#E8F6FF')

    line_alt, = ax_alt.plot([], [], color='#1EFF00', linewidth=2)
    ax_alt.set_title("Altitude (m)", color='#1EFF00', fontname='Consolas', fontsize=12)
    ax_alt.grid(color='#37474F', linestyle='-', linewidth=0.5)
    ax_alt.tick_params(colors='#E8F6FF')

    line_err, = ax_err.plot([], [], color='#FF4081', linewidth=2)
    ax_err.set_title("Heading Error (deg)", color='#FF4081', fontname='Consolas', fontsize=12)
    ax_err.grid(color='#37474F', linestyle='-', linewidth=0.5)
    ax_err.tick_params(colors='#E8F6FF')

    info_text_obj = ax_info.text(0.1, 0.5, "", fontsize=14, fontname='Consolas', color='#00FFFF', verticalalignment='center')

    def reset_anim(event):
        runner.reset_sim()
        line_map.set_data_3d([], [], [])
        line_alt.set_data([], [])
        line_err.set_data([], [])
        
        ax_map.set_xlim3d(-1100, 100)
        ax_map.set_ylim3d(-1100, 100)
        ax_map.set_zlim3d(0, 1200)
        ax_alt.set_xlim(0, 500)
        ax_alt.set_ylim(0, 1200)
        ax_err.set_xlim(0, 500)
        ax_err.set_ylim(-180, 180)

    btn_reset.on_clicked(reset_anim)
    reset_anim(None)

    def update_plot(frame):
        # Run a few simulation steps per visual frame to speed it up
        running = True
        for _ in range(5):
            running = runner.step()
            if not running:
                break
                
        if len(runner.history["x"]) == 0:
            return line_map, line_alt, line_err, info_text_obj
            
        line_map.set_data_3d(runner.history["x"], runner.history["y"], runner.history["alt"])
        
        t_steps = len(runner.history["alt"])
        if t_steps > ax_alt.get_xlim()[1]:
            ax_alt.set_xlim(0, t_steps + 500)
            ax_err.set_xlim(0, t_steps + 500)
            
        line_alt.set_data(range(t_steps), runner.history["alt"])
        line_err.set_data(range(t_steps), runner.history["heading_err"])
        
        phase_str = runner.dynamics.phase.name
        
        # Narada Terminal Output Style
        info_text = f"--- NARADA TELEMETRY ---\n\n"
        info_text += f"SIM TIME : {runner.sim_time:>8.1f} s\n"
        info_text += f"PHASE    : {phase_str:>10}\n"
        info_text += f"STATE    : {runner.history['state'][-1] if t_steps > 0 else 'N/A':>10}\n"
        info_text += f"ALTITUDE : {runner.dynamics.altitude:>8.1f} m\n"
        
        if not running:
            final_dist = math.hypot(runner.history["x"][-1] - runner.target_x, runner.history["y"][-1] - runner.target_y)
            info_text += f"\n--- MISSION COMPLETE ---\n"
            info_text += f"MISS DIST: {final_dist:>8.2f} m\n"
            if final_dist < 20.0:
                info_text += f"RESULT   : [SUCCESS]"
                info_text_obj.set_color('#1EFF00') # Lime
            else:
                info_text += f"RESULT   : [FAILURE]"
                info_text_obj.set_color('#FF4081') # Red
                
        info_text_obj.set_text(info_text)
        
        # update y limit for altitude if rocket goes higher
        if runner.dynamics.altitude > ax_alt.get_ylim()[1]:
            ax_alt.set_ylim(0, runner.dynamics.altitude + 200)

        # dynamic map scaling (3D)
        x_min, x_max = ax_map.get_xlim3d()
        y_min, y_max = ax_map.get_ylim3d()
        z_min, z_max = ax_map.get_zlim3d()
        curr_x, curr_y, curr_z = runner.dynamics.x, runner.dynamics.y, runner.dynamics.altitude
        if curr_x < x_min or curr_x > x_max or curr_y < y_min or curr_y > y_max:
            ax_map.set_xlim3d(min(x_min, curr_x - 100), max(x_max, curr_x + 100))
            ax_map.set_ylim3d(min(y_min, curr_y - 100), max(y_max, curr_y + 100))
            
        return line_map, line_alt, line_err, info_text_obj

    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    create_dashboard()
