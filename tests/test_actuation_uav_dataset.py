"""
tests/test_actuation_uav_dataset.py
=====================================
Downloads real PX4 ULog flight logs from HuggingFace UAV-SEAD dataset,
parses real attitude/baro/wind data, feeds it through the GARUD AGC
actuation algorithm and produces maximum visual + numerical outputs.

NOTE: These logs are indoor optical-flow flights (no GPS lat/lon).
We use real yaw, real baro altitude, real wind, and a synthetic NED
target to fully exercise the AGC -> mixer -> servo pipeline with live data.

Real data used per timestep:
  vehicle_local_position.yaw     -> actual heading (radians)
  vehicle_local_position.x,y     -> NED position (metres)
  vehicle_local_position.vx,vy   -> velocity (for speed display)
  sensor_baro.altitude           -> real barometric altitude
  wind_estimate.windspeed_N/E    -> real wind components

What this tests:
  heading_err -> AGC -> delta_a -> mixer -> servo_left, servo_right

Run:
    pip install pyulog matplotlib numpy
    python -m tests.test_actuation_uav_dataset
    python -m tests.test_actuation_uav_dataset --max-logs 8
    python -m tests.test_actuation_uav_dataset --target-ned-x 100 --target-ned-y 50
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

try:
    import numpy as np
    _NP = True
except ImportError:
    _NP = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    _PLT = True
except ImportError:
    _PLT = False
    print("[WARN] matplotlib not found -- pip install matplotlib")

try:
    from pyulog import ULog
    _ULOG = True
except ImportError:
    _ULOG = False
    print("[ERROR] pyulog not found -- pip install pyulog")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HF_BASE = (
    "https://huggingface.co/datasets/"
    "aykutkabaoglu/uav-flight-anomaly-dataset/resolve/main"
)

ULG_FILES = [
    "ulg_files/2018-06-04/18_01_59.ulg",
    "ulg_files/2018-06-04/18_05_54.ulg",
    "ulg_files/2018-06-04/18_09_57.ulg",
    "ulg_files/2018-06-04/18_13_16.ulg",
    "ulg_files/2018-06-04/18_15_21.ulg",
    "ulg_files/2018-06-04/18_17_58.ulg",
    "ulg_files/2018-06-04/18_21_53.ulg",
    "ulg_files/2018-06-04/18_25_07.ulg",
    "ulg_files/2018-06-04/18_33_02.ulg",
    "ulg_files/2018-06-04/18_36_56.ulg",
]

DATA_DIR   = Path("tests/data/uav_sead")
RESULT_DIR = Path("tests/results")

# AGC parameters -- must match gnc_process.py exactly
BASE_GAIN = 15.0
K_MAX     = 25.0
K_MIN     =  5.0
SIGMA     = 30.0

# Synthetic NED target (metres from local frame origin)
DEFAULT_TARGET_X =  100.0   # North
DEFAULT_TARGET_Y =   50.0   # East


# ---------------------------------------------------------------------------
# AGC + Mixer  (exact copy of gnc_process.py)
# ---------------------------------------------------------------------------

def agc(heading_err: float, altitude: float,
        wind_x: float, wind_y: float,
        target_bearing: float):
    cross   = -wind_x * math.sin(target_bearing) + wind_y * math.cos(target_bearing)
    K_wind  = BASE_GAIN + (K_MAX - K_MIN) * (abs(cross) / (abs(cross) + SIGMA))
    urgency = max(0.0, 1.0 - altitude / 300.0) if altitude > 10.0 else 0.0
    K_total = K_wind * (1.0 + 0.5 * urgency)
    delta_a = K_total * math.tanh(heading_err)
    delta_a = max(-30.0, min(30.0, delta_a))
    return delta_a, 5.0, K_wind, cross


def mix(delta_a: float, delta_s: float):
    left  = max(60.0, min(120.0, 90.0 + delta_s + delta_a))
    right = max(60.0, min(120.0, 90.0 + delta_s - delta_a))
    return left, right


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_ulg(remote_path: str, local_path: Path) -> bool:
    if local_path.exists() and local_path.stat().st_size > 1000:
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{HF_BASE}/{remote_path}"
    print(f"  [DL] {Path(remote_path).name}")
    try:
        urllib.request.urlretrieve(url, local_path)
        return local_path.stat().st_size > 1000
    except Exception as exc:
        print(f"  [FAIL] {Path(remote_path).name}: {exc}")
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# ULog parser  -- NED frame, real attitude + baro + wind
# ---------------------------------------------------------------------------

def _topic(ulog, name: str) -> Optional[Dict]:
    for d in ulog.data_list:
        if d.name == name:
            return d.data
    return None


def _interp(ts, vals, t: float) -> float:
    if not ts:
        return 0.0
    if t <= ts[0]:
        return float(vals[0])
    if t >= ts[-1]:
        return float(vals[-1])
    lo, hi = 0, len(ts) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if ts[mid] <= t:
            lo = mid
        else:
            hi = mid
    dt = ts[hi] - ts[lo]
    if dt == 0:
        return float(vals[lo])
    frac = (t - ts[lo]) / dt
    return float(vals[lo]) + frac * (float(vals[hi]) - float(vals[lo]))


def parse_ulg(path: Path, target_x: float, target_y: float) -> List[Dict]:
    """
    Parse a PX4 ULog.  Returns records with:
      time_s, ned_x, ned_y, alt_m, heading_rad,
      vx, vy, wind_x, wind_y,
      -- plus actuation outputs computed here:
      target_bearing, heading_err_deg, delta_a, delta_s,
      servo_left, servo_right, K_wind, cross_wind, dist_m
    """
    if not _ULOG:
        return []
    try:
        ulog = ULog(str(path))
    except Exception as exc:
        print(f"  [PARSE ERROR] {path.name}: {exc}")
        return []

    lpos = _topic(ulog, "vehicle_local_position")
    baro = _topic(ulog, "sensor_baro")
    wind = _topic(ulog, "wind_estimate")

    if lpos is None:
        print(f"  [SKIP] {path.name}: no vehicle_local_position topic")
        return []

    # Check we have a 'yaw' field (not all firmware versions include it)
    if "yaw" not in lpos:
        # Fallback: get yaw from vehicle_attitude quaternion
        att = _topic(ulog, "vehicle_attitude")
        if att is None or "q[0]" not in att:
            print(f"  [SKIP] {path.name}: no yaw or attitude quaternion")
            return []
        att_ts  = list(att["timestamp"])
        att_yaw = [
            math.atan2(
                2*(float(att["q[0]"][i])*float(att["q[3]"][i]) +
                   float(att["q[1]"][i])*float(att["q[2]"][i])),
                1 - 2*(float(att["q[2]"][i])**2 + float(att["q[3]"][i])**2)
            )
            for i in range(len(att_ts))
        ]
        lpos_ts  = list(lpos["timestamp"])
        yaw_vals = [_interp(att_ts, att_yaw, t) for t in lpos_ts]
    else:
        lpos_ts  = list(lpos["timestamp"])
        yaw_vals = [float(v) for v in lpos["yaw"]]

    # Position + velocity
    ned_x = [float(v) for v in lpos["x"]]
    ned_y = [float(v) for v in lpos["y"]]
    ned_z = [float(v) for v in lpos["z"]]
    vx_   = [float(v) for v in lpos.get("vx", [0.0]*len(lpos_ts))]
    vy_   = [float(v) for v in lpos.get("vy", [0.0]*len(lpos_ts))]

    # Baro altitude
    if baro is not None:
        baro_ts  = list(baro["timestamp"])
        baro_alt = [float(v) for v in baro["altitude"]]
    else:
        baro_ts, baro_alt = [], []

    # Wind (North = +x in ENU→NED convention PX4 uses)
    if wind is not None:
        wind_ts = list(wind["timestamp"])
        wind_n  = [float(v) for v in wind["windspeed_north"]]
        wind_e  = [float(v) for v in wind["windspeed_east"]]
    else:
        wind_ts, wind_n, wind_e = [], [], []

    # Subsample to max 2000 rows per log
    n    = len(lpos_ts)
    step = max(1, n // 2000)
    records = []
    t0 = lpos_ts[0]

    for i in range(0, n, step):
        t       = lpos_ts[i]
        heading = yaw_vals[i]
        x       = ned_x[i]
        y       = ned_y[i]
        z       = ned_z[i]

        alt = _interp(baro_ts, baro_alt, t) if baro_ts else max(0.0, -z)
        wx  = _interp(wind_ts, wind_e, t) if wind_ts else 0.0  # East component
        wy  = _interp(wind_ts, wind_n, t) if wind_ts else 0.0  # North component

        # Bearing to target in NED frame
        dx = target_x - x   # North
        dy = target_y - y   # East
        dist = math.hypot(dx, dy)
        # NED bearing: atan2(East, North) = atan2(dy, dx)
        target_bearing = math.atan2(dy, dx)
        heading_err    = wrap_pi(target_bearing - heading)

        delta_a, delta_s, K_wind, cross = agc(
            heading_err, alt, wx, wy, target_bearing)
        servo_left, servo_right = mix(delta_a, delta_s)

        records.append({
            "time_s":          (t - t0) / 1e6,
            "ned_x":           x,
            "ned_y":           y,
            "alt_m":           alt,
            "heading_rad":     heading,
            "heading_deg":     math.degrees(heading),
            "vx":              vx_[i],
            "vy":              vy_[i],
            "wind_x":          wx,
            "wind_y":          wy,
            "target_bearing":  target_bearing,
            "heading_err_rad": heading_err,
            "heading_err_deg": math.degrees(heading_err),
            "dist_m":          dist,
            "delta_a":         delta_a,
            "delta_s":         delta_s,
            "servo_left":      servo_left,
            "servo_right":     servo_right,
            "K_wind":          K_wind,
            "cross_wind":      cross,
        })

    return records


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(records: List[Dict]) -> Dict:
    if not records:
        return {"rows": 0}
    sl = [r["servo_left"]        for r in records]
    sr = [r["servo_right"]       for r in records]
    da = [r["delta_a"]           for r in records]
    he = [abs(r["heading_err_deg"]) for r in records]
    return {
        "rows":              len(records),
        "servo_L":           f"{min(sl):.1f}-{max(sl):.1f}",
        "servo_R":           f"{min(sr):.1f}-{max(sr):.1f}",
        "delta_a":           f"{min(da):.1f}-{max(da):.1f}",
        "mean_err_deg":      round(sum(he)/len(he), 2),
        "max_err_deg":       round(max(he), 2),
        "PASS_servo":        all(60 <= v <= 120 for v in sl+sr),
        "PASS_delta_a":      all(-30 <= v <= 30 for v in da),
        "PASS_no_nan":       all(math.isfinite(v) for v in da+sl+sr),
        "PASS_sign":         all(
            (r["delta_a"] > 0 and r["heading_err_deg"] > 0) or
            (r["delta_a"] < 0 and r["heading_err_deg"] < 0) or
            abs(r["heading_err_deg"]) < 1.0
            for r in records
        ),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#16213e")
    ax.set_title(title,  color="white",   fontsize=11, pad=8)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#333355")


def plot_all(all_records, log_names, target_x, target_y, result_dir):
    if not _PLT:
        print("[SKIP] matplotlib not available")
        return
    result_dir.mkdir(parents=True, exist_ok=True)
    flat   = [r for log in all_records for r in log]
    colors = plt.cm.tab10.colors

    he_all = [r["heading_err_deg"] for r in flat]
    da_all = [r["delta_a"]         for r in flat]
    sl_all = [r["servo_left"]      for r in flat]
    sr_all = [r["servo_right"]     for r in flat]
    kw_all = [r["K_wind"]          for r in flat]
    cw_all = [r["cross_wind"]      for r in flat]

    # ── 1. NED Trajectory coloured by heading error ─────────────────────
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#1a1a2e")
    errs = [abs(e) for e in he_all]
    norm = Normalize(vmin=0, vmax=max(errs) if errs else 1)
    sm   = ScalarMappable(cmap="RdYlGn_r", norm=norm)
    for i, (log, name) in enumerate(zip(all_records, log_names)):
        if not log:
            continue
        xs = [r["ned_x"] for r in log]
        ys = [r["ned_y"] for r in log]
        ce = [abs(r["heading_err_deg"]) for r in log]
        for j in range(len(log)-1):
            ax.plot([ys[j], ys[j+1]], [xs[j], xs[j+1]],
                    color=sm.to_rgba(ce[j]), lw=1.5, alpha=0.85)
        ax.plot(ys[0], xs[0], "o", color=colors[i%10], ms=7,
                label=name, zorder=5)
    ax.plot(target_y, target_x, "*", color="#FFD700", ms=20,
            zorder=10, label="NED Target")
    cb = plt.colorbar(sm, ax=ax)
    cb.set_label("Heading Error (deg)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
    _dark_ax(ax, "NED Trajectories -- Coloured by Heading Error",
             "East (m)", "North (m)")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white",
              loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(result_dir/"trajectory_map.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] trajectory_map.png")

    # ── 2. Heading error time series ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#1a1a2e")
    for i, (log, name) in enumerate(zip(all_records, log_names)):
        if not log:
            continue
        ts  = [r["time_s"] for r in log]
        err = [r["heading_err_deg"] for r in log]
        ax.plot(ts, err, lw=1.2, alpha=0.8, color=colors[i%10], label=name)
    ax.axhline(0,   color="#FFD700", lw=1.2, ls="--", label="Zero error")
    ax.axhline( 10, color="#FF4444", lw=0.8, ls=":", alpha=0.6)
    ax.axhline(-10, color="#FF4444", lw=0.8, ls=":", alpha=0.6, label="+-10deg band")
    _dark_ax(ax, "Heading Error vs Time -- Real PX4 Attitude Data",
             "Time (s)", "Heading Error (deg)")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", ncol=3)
    plt.tight_layout()
    plt.savefig(result_dir/"heading_error_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] heading_error_ts.png")

    # ── 3. Servo angles time series ─────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor="#1a1a2e",
                             sharex=False)
    for i, (log, name) in enumerate(zip(all_records, log_names)):
        if not log:
            continue
        ts = [r["time_s"] for r in log]
        axes[0].plot(ts, [r["servo_left"]  for r in log],
                     lw=1.2, alpha=0.8, color=colors[i%10],
                     label=name)
        axes[1].plot(ts, [r["servo_right"] for r in log],
                     lw=1.2, alpha=0.8, color=colors[i%10])
    for ax, lbl in zip(axes, ["Left Servo (deg)", "Right Servo (deg)"]):
        ax.axhline(90,  color="#FFD700", lw=1,   ls="--", label="Neutral 90deg")
        ax.axhline(60,  color="#FF4444", lw=0.8, ls=":", alpha=0.5, label="Limits")
        ax.axhline(120, color="#FF4444", lw=0.8, ls=":", alpha=0.5)
        ax.set_ylim(55, 125)
        _dark_ax(ax, lbl, "Time (s)", lbl)
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(result_dir/"servo_angles_ts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] servo_angles_ts.png")

    # ── 4. delta_a vs heading error (tanh nonlinearity check) ───────────
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#1a1a2e")
    ax.scatter(he_all[:10000], da_all[:10000], s=2, alpha=0.3,
               color="#4fc3f7", rasterized=True,
               label=f"Real data ({len(flat):,} pts)")
    if _NP:
        x_th = np.linspace(-180, 180, 400)
        y_b  = [max(-30, min(30, BASE_GAIN * math.tanh(math.radians(x)))) for x in x_th]
        y_m  = [max(-30, min(30, K_MAX     * math.tanh(math.radians(x)))) for x in x_th]
        ax.plot(x_th, y_b, color="#FFD700", lw=2.5,
                label=f"AGC base (K={BASE_GAIN})")
        ax.plot(x_th, y_m, color="#FF7043", lw=1.5, ls="--",
                label=f"AGC max (K={K_MAX}, strong wind)")
    ax.axhline( 30, color="#FF4444", lw=1, ls=":", alpha=0.7, label="+-30deg limit")
    ax.axhline(-30, color="#FF4444", lw=1, ls=":", alpha=0.7)
    ax.axvline(0,   color="#ffffff", lw=0.5, ls=":", alpha=0.3)
    ax.axhline(0,   color="#ffffff", lw=0.5, ls=":", alpha=0.3)
    _dark_ax(ax, "AGC: delta_a vs Heading Error  (tanh shape verification)",
             "Heading Error (deg)", "delta_a -- Aileron Output (deg)")
    ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(result_dir/"delta_a_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] delta_a_vs_error.png")

    # ── 5. K_wind adaptation ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#1a1a2e")
    ax.scatter(cw_all[:10000], kw_all[:10000], s=3, alpha=0.25,
               color="#AB47BC", rasterized=True, label="Real flight data")
    if _NP:
        cw_th = np.linspace(-15, 15, 300)
        kw_th = [BASE_GAIN + (K_MAX-K_MIN)*(abs(c)/(abs(c)+SIGMA))
                 for c in cw_th]
        ax.plot(cw_th, kw_th, color="#FFD700", lw=2.5,
                label="Theoretical K_wind curve")
    ax.axhline(BASE_GAIN, color="#4fc3f7", lw=1.5, ls="--",
               label=f"Base gain ({BASE_GAIN})")
    ax.axhline(K_MAX,     color="#FF4444", lw=1.5, ls=":",
               label=f"K_max ({K_MAX})")
    _dark_ax(ax, "Wind-Aware Gain Adaptation",
             "Crosswind Component (m/s)", "AGC Gain K_wind")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    plt.tight_layout()
    plt.savefig(result_dir/"agc_gain_vs_crosswind.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [OK] agc_gain_vs_crosswind.png")

    # ── 6. Servo histograms ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="#1a1a2e")
    for ax, vals, title, col in zip(
        axes,
        [sl_all, sr_all],
        ["Left Servo Distribution", "Right Servo Distribution"],
        ["#4fc3f7", "#FF7043"]
    ):
        ax.hist(vals, bins=60, color=col, edgecolor="#1a1a2e", alpha=0.9)
        ax.axvline(90,  color="#FFD700", lw=2,   ls="--", label="Neutral 90deg")
        ax.axvline(60,  color="#FF4444", lw=1.2, ls=":",  label="Limits [60-120]")
        ax.axvline(120, color="#FF4444", lw=1.2, ls=":")
        _dark_ax(ax, title, "Servo Angle (deg)", "Count")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.suptitle(
        f"Servo Angle Distributions  |  "
        f"{len(flat):,} real timesteps  |  {len(all_records)} logs",
        color="white", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(result_dir/"servo_histogram.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [OK] servo_histogram.png")

    # ── 7. Combined 3x3 dashboard ───────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), facecolor="#0f0f23")
    fig.suptitle(
        "GARUD Actuation Algorithm  --  Full Verification Dashboard\n"
        "UAV-SEAD Real PX4 Flight Data  |  "
        f"{len(flat):,} timesteps  |  {len(all_records)} logs",
        color="white", fontsize=15, y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.38)

    def mk(pos, title):
        ax = fig.add_subplot(gs[pos])
        _dark_ax(ax, title)
        return ax

    # [0,0] Trajectories
    ax = mk((0, 0), "NED Trajectories")
    for i, log in enumerate(all_records):
        if not log: continue
        ax.plot([r["ned_y"] for r in log],
                [r["ned_x"] for r in log],
                lw=1.2, alpha=0.7, color=colors[i%10])
    ax.plot(target_y, target_x, "*", color="#FFD700", ms=14, zorder=10)
    ax.set_xlabel("East (m)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("North (m)", color="#aaaaaa", fontsize=8)

    # [0,1] Heading error
    ax = mk((0, 1), "Heading Error (deg)")
    for i, log in enumerate(all_records[:8]):
        if not log: continue
        ax.plot([r["time_s"] for r in log],
                [r["heading_err_deg"] for r in log],
                lw=0.9, alpha=0.8, color=colors[i%10])
    ax.axhline(0, color="#FFD700", lw=1, ls="--")
    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)

    # [0,2] Heading (raw)
    ax = mk((0, 2), "Real Heading (deg)")
    for i, log in enumerate(all_records[:8]):
        if not log: continue
        ax.plot([r["time_s"] for r in log],
                [r["heading_deg"] for r in log],
                lw=0.9, alpha=0.8, color=colors[i%10])
    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)

    # [1,0] delta_a scatter
    ax = mk((1, 0), "delta_a vs Error (tanh check)")
    ax.scatter(he_all[:6000], da_all[:6000], s=1.5, alpha=0.3,
               color="#4fc3f7", rasterized=True)
    if _NP:
        x_t = np.linspace(-180, 180, 300)
        y_t = [max(-30, min(30, BASE_GAIN*math.tanh(math.radians(x)))) for x in x_t]
        ax.plot(x_t, y_t, color="#FFD700", lw=1.8, label="AGC tanh")
    ax.axhline( 30, color="#FF4444", lw=0.8, ls=":", alpha=0.7)
    ax.axhline(-30, color="#FF4444", lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("Heading Error (deg)", color="#aaaaaa", fontsize=8)
    ax.set_ylabel("delta_a (deg)",       color="#aaaaaa", fontsize=8)
    if _NP:
        ax.legend(facecolor="#16213e", labelcolor="white", fontsize=8)

    # [1,1] Left servo time series
    ax = mk((1, 1), "Left Servo (deg)")
    for i, log in enumerate(all_records[:8]):
        if not log: continue
        ax.plot([r["time_s"] for r in log],
                [r["servo_left"] for r in log],
                lw=0.9, alpha=0.8, color=colors[i%10])
    ax.axhline(90, color="#FFD700", lw=1, ls="--")
    ax.axhline(60,  color="#FF4444", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(120, color="#FF4444", lw=0.6, ls=":", alpha=0.6)
    ax.set_ylim(55, 125)
    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)

    # [1,2] Right servo time series
    ax = mk((1, 2), "Right Servo (deg)")
    for i, log in enumerate(all_records[:8]):
        if not log: continue
        ax.plot([r["time_s"] for r in log],
                [r["servo_right"] for r in log],
                lw=0.9, alpha=0.8, color=colors[i%10])
    ax.axhline(90, color="#FFD700", lw=1, ls="--")
    ax.axhline(60,  color="#FF4444", lw=0.6, ls=":", alpha=0.6)
    ax.axhline(120, color="#FF4444", lw=0.6, ls=":", alpha=0.6)
    ax.set_ylim(55, 125)
    ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=8)

    # [2,0] Left servo histogram
    ax = mk((2, 0), "Left Servo Histogram")
    ax.hist(sl_all, bins=50, color="#4fc3f7", edgecolor="#1a1a2e", alpha=0.9)
    ax.axvline(90, color="#FFD700", lw=1.5, ls="--")
    ax.axvline(60,  color="#FF4444", lw=1, ls=":", alpha=0.7)
    ax.axvline(120, color="#FF4444", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Servo Angle (deg)", color="#aaaaaa", fontsize=8)

    # [2,1] Right servo histogram
    ax = mk((2, 1), "Right Servo Histogram")
    ax.hist(sr_all, bins=50, color="#FF7043", edgecolor="#1a1a2e", alpha=0.9)
    ax.axvline(90, color="#FFD700", lw=1.5, ls="--")
    ax.axvline(60,  color="#FF4444", lw=1, ls=":", alpha=0.7)
    ax.axvline(120, color="#FF4444", lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("Servo Angle (deg)", color="#aaaaaa", fontsize=8)

    # [2,2] Summary text
    ax = mk((2, 2), "")
    ax.axis("off")
    checks   = [verify(l) for l in all_records if l]
    total    = sum(c["rows"] for c in checks)
    ps       = all(c["PASS_servo"]   for c in checks)
    pd       = all(c["PASS_delta_a"] for c in checks)
    pn       = all(c["PASS_no_nan"]  for c in checks)
    pg       = all(c["PASS_sign"]    for c in checks)
    me       = sum(c["mean_err_deg"] for c in checks) / max(len(checks), 1)
    all_pass = ps and pd and pn and pg
    col      = "#00FF88" if all_pass else "#FF4444"
    result   = "ALL PASS" if all_pass else "CHECK FAILS"
    txt = (
        f"RESULT: {result}\n"
        f"{'='*22}\n"
        f"Logs processed : {len(checks)}\n"
        f"Total rows     : {total:,}\n\n"
        f"Servo in [60,120]  : {'PASS' if ps else 'FAIL'}\n"
        f"delta_a in [-30,30]: {'PASS' if pd else 'FAIL'}\n"
        f"No NaN/Inf         : {'PASS' if pn else 'FAIL'}\n"
        f"Correct sign       : {'PASS' if pg else 'FAIL'}\n\n"
        f"Mean |heading err| : {me:.1f} deg\n\n"
        f"AGC params:\n"
        f"  BASE_GAIN = {BASE_GAIN}\n"
        f"  K_MAX     = {K_MAX}\n"
        f"  K_MIN     = {K_MIN}\n"
        f"  SIGMA     = {SIGMA}"
    )
    ax.text(0.05, 0.97, txt, transform=ax.transAxes,
            fontsize=9.5, va="top", color=col, fontfamily="monospace")

    plt.savefig(result_dir/"combined_dashboard.png", dpi=150,
                bbox_inches="tight")
    plt.close()
    print("  [OK] combined_dashboard.png")


# ---------------------------------------------------------------------------
# Save CSV + report
# ---------------------------------------------------------------------------

def save_csv(all_records, log_names, result_dir):
    result_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "log", "time_s", "ned_x", "ned_y", "alt_m",
        "heading_rad", "heading_deg", "heading_err_deg",
        "target_bearing", "dist_m",
        "delta_a", "delta_s", "servo_left", "servo_right",
        "K_wind", "cross_wind", "wind_x", "wind_y", "vx", "vy",
    ]
    with open(result_dir/"actuation_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for log, name in zip(all_records, log_names):
            for r in log:
                row = {k: r.get(k, "") for k in fields}
                row["log"] = name
                w.writerow(row)
    total = sum(len(l) for l in all_records)
    print(f"  [OK] actuation_results.csv  ({total:,} rows)")


def save_report(all_records, log_names, result_dir, target_x, target_y):
    checks = [verify(l) for l in all_records]
    lines  = [
        "="*76,
        "GARUD Actuation Algorithm -- Full Verification Report",
        f"Dataset : HuggingFace UAV-SEAD (real PX4 indoor flights)",
        f"Target  : NED ({target_x:.1f}m N, {target_y:.1f}m E)",
        f"Date    : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "="*76, "",
        f"{'Log':<28} {'Rows':>5} {'Servo':>6} {'dA':>6} "
        f"{'NaN':>6} {'Sign':>6} {'MeanErr':>9}",
        "-"*76,
    ]
    all_pass = True
    for c, name in zip(checks, log_names):
        ps = "PASS" if c.get("PASS_servo")   else "FAIL"
        pd = "PASS" if c.get("PASS_delta_a") else "FAIL"
        pn = "PASS" if c.get("PASS_no_nan")  else "FAIL"
        pg = "PASS" if c.get("PASS_sign")    else "FAIL"
        if "FAIL" in [ps, pd, pn, pg]:
            all_pass = False
        lines.append(
            f"{name[:28]:<28} {c.get('rows',0):>5} "
            f"{ps:>6} {pd:>6} {pn:>6} {pg:>6} "
            f"{c.get('mean_err_deg',0):>8.2f}deg"
        )
    lines += [
        "-"*76, "",
        f"OVERALL RESULT: {'ALL PASS' if all_pass else 'SOME CHECKS FAILED'}", "",
        "Verification checks:",
        "  servo    : servo_left and servo_right within [60, 120] deg",
        "  delta_a  : AGC output within [-30, +30] deg",
        "  no_nan   : no NaN or Inf in any output",
        "  sign     : delta_a sign consistent with heading error sign",
        "",
        "Outputs saved to: tests/results/",
        "  trajectory_map.png      -- NED flight paths, error coloured",
        "  heading_error_ts.png    -- heading error over time",
        "  servo_angles_ts.png     -- left/right servo over time",
        "  delta_a_vs_error.png    -- tanh nonlinearity verification",
        "  agc_gain_vs_crosswind.png -- wind-adaptive gain",
        "  servo_histogram.png     -- servo angle distributions",
        "  combined_dashboard.png  -- 3x3 full overview",
        "  actuation_results.csv   -- all computed values",
        "  summary_report.txt      -- this file",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(result_dir/"summary_report.txt", "w") as f:
        f.write(report)
    print(f"\n  [OK] summary_report.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-logs",      type=int,   default=10)
    ap.add_argument("--target-ned-x",  type=float, default=DEFAULT_TARGET_X,
                    help="Target North offset (m) from local frame origin")
    ap.add_argument("--target-ned-y",  type=float, default=DEFAULT_TARGET_Y,
                    help="Target East offset (m) from local frame origin")
    args = ap.parse_args()

    if not _ULOG:
        print("[ERROR] pip install pyulog")
        sys.exit(1)

    print("\n" + "="*62)
    print("  GARUD Actuation Algorithm Test")
    print("  Dataset: UAV-SEAD  (HuggingFace / real PX4 flights)")
    print("="*62)
    print(f"  Target  : NED ({args.target_ned_x:.0f}m N, {args.target_ned_y:.0f}m E)")
    print(f"  Max logs: {args.max_logs}")
    print()

    # 1. Download
    print("[1/4] Downloading ULog files ...")
    selected   = ULG_FILES[:args.max_logs]
    downloaded = []
    for remote in selected:
        local = DATA_DIR / Path(remote).name
        if download_ulg(remote, local):
            downloaded.append(local)
    if not downloaded:
        print("[ERROR] No files downloaded.")
        sys.exit(1)
    print(f"  -> {len(downloaded)} files ready.\n")

    # 2. Parse + run AGC
    print("[2/4] Parsing ULogs and running AGC ...")
    all_records, log_names = [], []
    for path in downloaded:
        records = parse_ulg(path, args.target_ned_x, args.target_ned_y)
        if not records:
            continue
        all_records.append(records)
        log_names.append(path.stem[:28])
        me = sum(abs(r["heading_err_deg"]) for r in records) / len(records)
        v  = verify(records)
        ok = "PASS" if v["PASS_servo"] and v["PASS_delta_a"] else "FAIL"
        print(f"  {path.name:<32} {len(records):>5} rows  "
              f"mean_err={me:5.1f}deg  [{ok}]")

    if not all_records:
        print("[ERROR] No usable data parsed.")
        sys.exit(1)

    total = sum(len(l) for l in all_records)
    print(f"\n  -> {total:,} total timesteps / {len(all_records)} logs\n")

    # 3. CSV
    print("[3/4] Saving CSV ...")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(all_records, log_names, RESULT_DIR)

    # 4. Plots + report
    print("\n[4/4] Generating plots ...")
    plot_all(all_records, log_names,
             args.target_ned_x, args.target_ned_y, RESULT_DIR)
    save_report(all_records, log_names, RESULT_DIR,
                args.target_ned_x, args.target_ned_y)

    print(f"\n{'='*62}")
    print("  Done! Open tests/results/ to view all outputs.")
    print(f"  Files:")
    for f in sorted(RESULT_DIR.glob("*")):
        size = f.stat().st_size
        print(f"    {f.name:<35} {size//1024:>5} KB")


if __name__ == "__main__":
    main()
