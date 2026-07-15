import logging
import math
from collections import deque

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.align import Align
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

class DashboardLogHandler(logging.Handler):
    def __init__(self, capacity=12):
        super().__init__()
        self.logs = deque(maxlen=capacity)
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
        except Exception:
            self.handleError(record)

class FlightDashboard:
    """
    Rich-based Terminal Dashboard for the CAN-7U-SAT Flight Computer.
    """
    def __init__(self, enabled=True):
        self.enabled = enabled and RICH_AVAILABLE
        self.live = None
        self.layout = None
        self.log_handler = None
        
        if self.enabled:
            # Route logs to the dashboard
            self.log_handler = DashboardLogHandler(capacity=12)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%H:%M:%S')
            self.log_handler.setFormatter(formatter)
            logging.getLogger().addHandler(self.log_handler)
            
            # Setup Layout
            self.layout = Layout()
            self.layout.split_column(
                Layout(name="upper", ratio=3),
                Layout(name="logs", ratio=1, minimum_size=14)
            )
            self.layout["upper"].split_row(
                Layout(name="status_col", ratio=1),
                Layout(name="obs", ratio=1)
            )
            self.layout["status_col"].split_column(
                Layout(name="status", ratio=1),
                Layout(name="sensors", ratio=1)
            )

    def start(self):
        if self.enabled:
            self.live = Live(self.layout, refresh_per_second=10)
            self.live.start()
            
    def stop(self):
        if self.enabled and self.live:
            self.live.stop()
            
    def update(self, state_name, controller, baro_alt, dist, 
               roll, pitch, yaw, gps_speed, gps_heading, 
               wind_speed, wind_dir,
               left_servo, right_servo, delta_a, delta_s,
               obs=None):
        if not self.enabled:
            return

        # 1. Status Panel
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column(justify="right", style="cyan")
        status_table.add_column(style="magenta", justify="left")
        status_table.add_row("Flight State:", state_name)
        status_table.add_row("Controller:", controller)
        status_table.add_row("Altitude:", f"{baro_alt:.1f} m")
        status_table.add_row("Dist to Tgt:", f"{dist:.1f} m")
        
        self.layout["status"].update(Panel(Align.center(status_table, vertical="middle"), title="[bold]Flight Status"))
        
        # 2. Sensors Panel
        sensor_table = Table.grid(padding=(0, 2))
        sensor_table.add_column(justify="right", style="cyan")
        sensor_table.add_column(style="yellow")
        sensor_table.add_row("Roll / Pitch:", f"{math.degrees(roll):.1f}° / {math.degrees(pitch):.1f}°")
        sensor_table.add_row("Yaw (Mag):", f"{math.degrees(yaw):.1f}°")
        sensor_table.add_row("GPS Speed:", f"{gps_speed:.1f} m/s")
        sensor_table.add_row("GPS COG:", f"{math.degrees(gps_heading):.1f}°")
        sensor_table.add_row("Wind Speed:", f"{wind_speed:.1f} m/s")
        sensor_table.add_row("Wind Dir:", f"{math.degrees(wind_dir):.1f}°")
        sensor_table.add_row("Left Servo:", f"{left_servo:.1f}°")
        sensor_table.add_row("Right Servo:", f"{right_servo:.1f}°")
        sensor_table.add_row("Cmd (δa/δs):", f"{delta_a:.1f}° / {delta_s:.1f}°")
        
        self.layout["sensors"].update(Panel(Align.center(sensor_table, vertical="middle"), title="[bold]Sensors & Actuators"))
        
        # 3. Obs Panel
        if obs is not None:
            obs_table = Table(show_header=True, header_style="bold magenta", padding=(0, 1))
            obs_table.add_column("Idx")
            obs_table.add_column("Signal")
            obs_table.add_column("Value", justify="right")
            
            labels = ["sin(hdg)", "cos(hdg)", "dist/1k", "alt_exc/1k", "w_spd/10", "sin(w_dir)", "cos(w_dir)",
                      "pitch/0.5", "roll/0.5", "yaw_rt/0.5", "prev_da", "prev_ds", "sin(trk)", "cos(trk)", 
                      "lat_drift", "tti/200"]
            for i, (l, v) in enumerate(zip(labels, obs)):
                obs_table.add_row(str(i), l, f"{v:.3f}")
                
            self.layout["obs"].update(Panel(Align.center(obs_table, vertical="middle"), title="[bold]RL Observation Space (16D)"))
        else:
            self.layout["obs"].update(Panel(Align.center(Text("RL Not Active"), vertical="middle"), title="[bold]RL Observation Space (16D)"))
            
        # 4. Logs
        log_text = Text("\n".join(self.log_handler.logs))
        self.layout["logs"].update(Panel(log_text, title="[bold]System Logs"))
