# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
tests/test_actuation_multi_dataset.py
======================================
Tests the GARUD AGC actuation algorithm against TWO real datasets:

  DATASET A — UAV-SEAD (HuggingFace, PX4 ULog binary)
    aykutkabaoglu/uav-flight-anomaly-dataset
    * 200+ date folders spanning 2018-2022
    * Samples logs from 5 different dates (previously only used 2018-06-04)
    * Parses: yaw, NED position, baro altitude, wind estimate
    * Format: .ulg binary (via pyulog)

  DATASET B — Outdoor UAV GPS Trajectories (HuggingFace, CSV)
    riotu-lab/os-rfodg-outdoor-uav-synthetic-dataset-taif-saudi-arabia
    * 7 outdoor trajectory CSV files over real DEM terrain (Taif, Saudi Arabia)
    * Has: real GPS lat/lon/altitude, quaternion orientation (-> yaw), velocities
    * Format: CSV (no special library needed)

What is tested (same 4 checks for both datasets):
  heading_err -> AGC -> delta_a -> mixer -> servo_left, servo_right
  1. Servo in [60, 120] deg
  2. delta_a in [-30, +30] deg
  3. No NaN / Inf
  4. Sign of delta_a consistent with sign of heading_error

Run:
    python -m tests.test_actuation_multi_dataset
    python -m tests.test_actuation_multi_dataset --max-ulg 5 --max-csv 7
"""

import argparse
import csv
import io
import math
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── optional imports ─────────────────────────────────────────────────────────
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

try:
    from pyulog import ULog
    _ULOG = True
except ImportError:
    _ULOG = False

# ── paths / constants ────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
DATA_DIR   = HERE / "data" / "multi_dataset"
RESULTS    = HERE / "results_multi"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

HF_BASE_SEAD = (
    "https://huggingface.co/datasets/"
    "aykutkabaoglu/uav-flight-anomaly-dataset/resolve/main"
)
HF_BASE_RIOTU = (
    "https://huggingface.co/datasets/"
    "riotu-lab/os-rfodg-outdoor-uav-synthetic-dataset-taif-saudi-arabia"
    "/resolve/main"
)

# Sample 5 different dates from UAV-SEAD (variety of conditions)
SEAD_DATES = [
    ("2018-06-04", ["18_01_59.ulg", "18_09_57.ulg"]),   # already tested
    ("2018-09-05", None),   # will auto-discover
    ("2019-01-04", None),
    ("2020-02-14", None),
    ("2022-09-07", None),
]

# All 7 outdoor GPS trajectory CSVs from riotu-lab
RIOTU_FILES = [f"traj_{i}_taif_map_data.csv" for i in range(1, 8)]

# AGC parameters (matches flight_computer.py)
BASE_GAIN  = 15.0
K_MAX      = 25.0
K_MIN      =  5.0
SIGMA      = 30.0
DELTA_A_CLAMP = 30.0
SERVO_MIN  = 60.0
SERVO_MAX  = 120.0


# ── helpers ──────────────────────────────────────────────────────────────────
def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_yaw(ox: float, oy: float, oz: float, ow: float) -> float:
    """Convert quaternion to yaw (rotation about Z axis) in radians."""
    siny_cosp = 2.0 * (ow * oz + ox * oy)
    cosy_cosp = 1.0 - 2.0 * (oy * oy + oz * oz)
    return math.atan2(siny_cosp, cosy_cosp)


def agc_step(heading_err: float, wind_n: float, wind_e: float,
             altitude: float) -> Tuple[float, float, float, float]:
    """
    Run one step of the AGC -> mixer pipeline.
    Returns: (delta_a, servo_left, servo_right, K_wind)
    """
    crosswind = abs(wind_e * math.cos(heading_err) - wind_n * math.sin(heading_err))
    K_wind    = K_MIN + (K_MAX - K_MIN) * crosswind / (crosswind + SIGMA)
    urgency   = min(1.0, max(0.0, 1.0 - altitude / 600.0))
    K_total   = BASE_GAIN * K_wind * (1.0 + 0.5 * urgency)

    delta_a = K_total * math.tanh(math.radians(heading_err))
    delta_a = max(-DELTA_A_CLAMP, min(DELTA_A_CLAMP, delta_a))

    delta_s   = 0.0
    srv_left  = max(SERVO_MIN, min(SERVO_MAX, 90.0 + delta_s - delta_a))
    srv_right = max(SERVO_MIN, min(SERVO_MAX, 90.0 + delta_s + delta_a))
    return delta_a, srv_left, srv_right, K_wind


def download(url: str, dest: Path, label: str = "") -> bool:
    if dest.exists():
        return True
    print(f"  [DL] {label or dest.name}")
    try:
        urllib.request.urlretrieve(url, dest)
        return True
    except Exception as e:
        print(f"  [SKIP] {dest.name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def discover_sead_files(date: str, max_files: int = 3) -> List[str]:
    """Ask HF API for the first few .ulg files in a date folder."""
    api_url = (
        f"https://huggingface.co/api/datasets/"
        f"aykutkabaoglu/uav-flight-anomaly-dataset/tree/main/ulg_files/{date}"
    )
    try:
        with urllib.request.urlopen(api_url, timeout=10) as r:
            import json
            entries = json.loads(r.read().decode())
        files = [e["path"].split("/")[-1] for e in entries
                 if e.get("type") == "file" and e["path"].endswith(".ulg")]
        return files[:max_files]
    except Exception:
        return []


# ── DATASET A: UAV-SEAD ULog parser ─────────────────────────────────────────
def parse_ulg(path: Path, target_ned_x: float = 100.0,
              target_ned_y: float = 50.0) -> Optional[List[dict]]:
    if not _ULOG:
        print("  [ERROR] pyulog not installed. pip install pyulog")
        return None
    try:
        ulog = ULog(str(path), disable_str_exceptions=True)
    except Exception as e:
        print(f"  [ERROR] ULog parse failed {path.name}: {e}")
        return None

    def get_topic(name: str, fields: List[str]) -> Optional[dict]:
        for d in ulog.data_list:
            if d.name == name:
                if all(f in d.data for f in fields):
                    return {f: d.data[f] for f in ["timestamp"] + fields
                            if f in d.data}
        return None

    pos  = get_topic("vehicle_local_position", ["x", "y", "yaw"])
    baro = get_topic("sensor_baro", ["altitude"])
    wind = get_topic("wind_estimate", ["windspeed_north", "windspeed_east"])
    if pos is None:
        return None

    rows = []
    n = len(pos["timestamp"])
    for i in range(n):
        try:
            yaw = float(pos["yaw"][i])
            px  = float(pos["x"][i])
            py  = float(pos["y"][i])
            alt = float(baro["altitude"][i]) if (baro and i < len(baro["altitude"])) else 130.0
            wn  = float(wind["windspeed_north"][i]) if (wind and i < len(wind["windspeed_north"])) else 0.0
            we  = float(wind["windspeed_east"][i]) if (wind and i < len(wind["windspeed_east"])) else 0.0

            bearing = math.atan2(target_ned_y - py, target_ned_x - px)
            h_err   = math.degrees(wrap_pi(bearing - yaw))
            da, sl, sr, kw = agc_step(h_err, wn, we, alt)

            rows.append({
                "log": path.name, "dataset": "UAV-SEAD",
                "heading_err": h_err, "delta_a": da,
                "servo_left": sl, "servo_right": sr,
                "altitude": alt, "wind_n": wn, "wind_e": we,
                "K_wind": kw, "x": px, "y": py, "yaw_deg": math.degrees(yaw),
            })
        except (IndexError, ValueError, ZeroDivisionError):
            continue
    return rows if rows else None


# ── DATASET B: riotu-lab CSV parser ─────────────────────────────────────────
def parse_riotu_csv(path: Path,
                    target_lat: float = 21.25,
                    target_lon: float = 40.23) -> Optional[List[dict]]:
    """
    CSV columns:
      timestamp, image_name, tx, ty, tz,
      vel_x, vel_y, vel_z,
      orientation_x, orientation_y, orientation_z, orientation_w,
      angular_vel_x/y/z, linear_acc_x/y/z,
      latitude, longitude, altitude, lidar_range, AMSL
    """
    rows = []
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ox  = float(row["orientation_x"])
                    oy  = float(row["orientation_y"])
                    oz  = float(row["orientation_z"])
                    ow  = float(row["orientation_w"])
                    lat = float(row["latitude"])
                    lon = float(row["longitude"])
                    alt = float(row["altitude"])

                    # Derive yaw from quaternion
                    yaw = quat_to_yaw(ox, oy, oz, ow)

                    # GPS bearing to a fixed target near the trajectory center
                    dlat = (target_lat - lat) * 111320.0          # metres North
                    dlon = (target_lon - lon) * 111320.0 * math.cos(math.radians(lat))
                    bearing = math.atan2(dlon, dlat)
                    h_err = math.degrees(wrap_pi(bearing - yaw))

                    # No wind in this dataset — use 0 (tests calm-air servo behaviour)
                    da, sl, sr, kw = agc_step(h_err, 0.0, 0.0, alt)

                    rows.append({
                        "log": path.name, "dataset": "riotu-GPS",
                        "heading_err": h_err, "delta_a": da,
                        "servo_left": sl, "servo_right": sr,
                        "altitude": alt, "wind_n": 0.0, "wind_e": 0.0,
                        "K_wind": kw,
                        "x": float(row["tx"]), "y": float(row["ty"]),
                        "yaw_deg": math.degrees(yaw),
                        "latitude": lat, "longitude": lon,
                    })
                except (KeyError, ValueError):
                    continue
    except Exception as e:
        print(f"  [ERROR] CSV parse {path.name}: {e}")
        return None
    return rows if rows else None


# ── verification ─────────────────────────────────────────────────────────────
def verify(rows: List[dict], log_name: str) -> dict:
    srv_ok  = all(SERVO_MIN <= r["servo_left"]  <= SERVO_MAX and
                  SERVO_MIN <= r["servo_right"] <= SERVO_MAX for r in rows)
    da_ok   = all(abs(r["delta_a"]) <= DELTA_A_CLAMP for r in rows)
    nan_ok  = all(math.isfinite(r["delta_a"]) and
                  math.isfinite(r["servo_left"]) for r in rows)
    sign_ok = all(
        (r["heading_err"] >= 0) == (r["delta_a"] >= 0)
        for r in rows if abs(r["heading_err"]) > 1.0
    )
    mean_err = sum(abs(r["heading_err"]) for r in rows) / len(rows)
    return {
        "log": log_name,
        "rows": len(rows),
        "servo": "PASS" if srv_ok  else "FAIL",
        "delta_a": "PASS" if da_ok  else "FAIL",
        "no_nan": "PASS" if nan_ok  else "FAIL",
        "sign":   "PASS" if sign_ok else "FAIL",
        "mean_err": mean_err,
        "overall": "PASS" if all([srv_ok, da_ok, nan_ok, sign_ok]) else "FAIL",
    }


# ── plotting ─────────────────────────────────────────────────────────────────
def make_plots(all_rows: List[dict], sead_results: list, riotu_results: list):
    if not _PLT or not _NP:
        print("  [SKIP] matplotlib/numpy not available for plotting")
        return

    # Separate datasets
    sead_rows  = [r for r in all_rows if r["dataset"] == "UAV-SEAD"]
    riotu_rows = [r for r in all_rows if r["dataset"] == "riotu-GPS"]

    fig = plt.figure(figsize=(20, 24))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(4, 3, figure=fig,
                            hspace=0.45, wspace=0.35,
                            top=0.93, bottom=0.04,
                            left=0.06, right=0.97)

    NAVY = "#002764"; BLUE = "#006EC7"; GREEN = "#28a745"
    RED  = "#dc3545"; GOLD = "#ffc107"; GREY = "#6c757d"
    TEXT = "#E6EDF3"; GRID = "#21262D"

    def ax_style(ax, title, xlabel, ylabel):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(color=GRID, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor(GREY)

    # ── Row 0: UAV-SEAD Overview ─────────────────────────────────────────────
    # 0,0 — heading error time-series (SEAD)
    ax = fig.add_subplot(gs[0, 0])
    if sead_rows:
        idx = list(range(len(sead_rows)))
        errs = [r["heading_err"] for r in sead_rows]
        ax.scatter(idx, errs, s=0.4, c=BLUE, alpha=0.5)
        ax.axhline(0, color=GOLD, lw=0.8, ls="--")
    ax_style(ax, "UAV-SEAD: Heading Error vs Timestep",
             "Timestep", "Heading Error (deg)")

    # 0,1 — servo angles (SEAD)
    ax = fig.add_subplot(gs[0, 1])
    if sead_rows:
        idx = list(range(len(sead_rows)))
        ax.scatter(idx, [r["servo_left"]  for r in sead_rows], s=0.4, c=BLUE,  alpha=0.4, label="Left")
        ax.scatter(idx, [r["servo_right"] for r in sead_rows], s=0.4, c=GREEN, alpha=0.4, label="Right")
        ax.axhline(60,  color=RED,  lw=0.8, ls="--")
        ax.axhline(120, color=RED,  lw=0.8, ls="--")
        ax.axhline(90,  color=GOLD, lw=0.8, ls=":")
        ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "UAV-SEAD: Servo Angles", "Timestep", "Servo Angle (deg)")

    # 0,2 — delta_a vs heading error (SEAD)
    ax = fig.add_subplot(gs[0, 2])
    if sead_rows:
        errs = np.array([r["heading_err"] for r in sead_rows])
        das  = np.array([r["delta_a"]     for r in sead_rows])
        ax.scatter(errs, das, s=0.4, c=BLUE, alpha=0.3)
        xe = np.linspace(-180, 180, 300)
        ax.plot(xe, DELTA_A_CLAMP * np.tanh(np.radians(xe) * BASE_GAIN / DELTA_A_CLAMP),
                color=GOLD, lw=1.5, label="tanh curve")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "UAV-SEAD: delta_a vs Heading Error",
             "Heading Error (deg)", "delta_a (deg)")

    # ── Row 1: riotu-lab GPS Overview ───────────────────────────────────────
    # 1,0 — GPS trajectory (riotu)
    ax = fig.add_subplot(gs[1, 0])
    if riotu_rows and "latitude" in riotu_rows[0]:
        lats = [r["latitude"]  for r in riotu_rows]
        lons = [r["longitude"] for r in riotu_rows]
        errs = [abs(r["heading_err"]) for r in riotu_rows]
        cmap = plt.cm.RdYlGn_r
        sc = ax.scatter(lons, lats, c=errs, cmap=cmap, s=1.5, alpha=0.6,
                        vmin=0, vmax=180)
        plt.colorbar(sc, ax=ax).ax.yaxis.label.set_color(TEXT)
        # Mark target
        ax.scatter([40.23], [21.25], c="yellow", s=80, marker="*", zorder=5)
    ax_style(ax, "riotu-GPS: GPS Trajectories (7 flights)\n[colour=|heading error|]",
             "Longitude", "Latitude")

    # 1,1 — heading error over timestep (riotu)
    ax = fig.add_subplot(gs[1, 1])
    if riotu_rows:
        errs = [r["heading_err"] for r in riotu_rows]
        ax.scatter(range(len(errs)), errs, s=0.4, c=GREEN, alpha=0.4)
        ax.axhline(0, color=GOLD, lw=0.8, ls="--")
    ax_style(ax, "riotu-GPS: Heading Error vs Timestep",
             "Timestep", "Heading Error (deg)")

    # 1,2 — delta_a vs heading error (riotu)
    ax = fig.add_subplot(gs[1, 2])
    if riotu_rows:
        errs = np.array([r["heading_err"] for r in riotu_rows])
        das  = np.array([r["delta_a"]     for r in riotu_rows])
        ax.scatter(errs, das, s=0.4, c=GREEN, alpha=0.3)
        xe = np.linspace(-180, 180, 300)
        ax.plot(xe, DELTA_A_CLAMP * np.tanh(np.radians(xe) * BASE_GAIN / DELTA_A_CLAMP),
                color=GOLD, lw=1.5, label="tanh curve")
        ax.axhline(0, color=GREY, lw=0.5)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "riotu-GPS: delta_a vs Heading Error",
             "Heading Error (deg)", "delta_a (deg)")

    # ── Row 2: Combined comparisons ──────────────────────────────────────────
    # 2,0 — Servo histogram (both datasets)
    ax = fig.add_subplot(gs[2, 0])
    if sead_rows:
        sls = [r["servo_left"] for r in sead_rows]
        ax.hist(sls, bins=40, color=BLUE,  alpha=0.6, label="SEAD Left",  density=True)
    if riotu_rows:
        sls = [r["servo_left"] for r in riotu_rows]
        ax.hist(sls, bins=40, color=GREEN, alpha=0.6, label="GPS Left",   density=True)
    ax.axvline(60,  color=RED,  lw=1, ls="--", label="Limits")
    ax.axvline(120, color=RED,  lw=1, ls="--")
    ax.axvline(90,  color=GOLD, lw=1, ls=":",  label="Neutral")
    ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "Left Servo Distribution (Both Datasets)",
             "Servo Angle (deg)", "Density")

    # 2,1 — K_wind adaptive gain (both datasets)
    ax = fig.add_subplot(gs[2, 1])
    if sead_rows:
        ax.scatter(range(len(sead_rows)),  [r["K_wind"] for r in sead_rows],
                   s=0.5, c=BLUE,  alpha=0.4, label="SEAD (no wind)")
    if riotu_rows:
        ax.scatter(range(len(riotu_rows)), [r["K_wind"] for r in riotu_rows],
                   s=0.5, c=GREEN, alpha=0.4, label="GPS (no wind)")
    ax.axhline(BASE_GAIN, color=GOLD, lw=1, ls="--", label=f"BASE={BASE_GAIN}")
    ax.axhline(K_MIN,     color=RED,  lw=0.8, ls=":")
    ax.axhline(K_MAX,     color=RED,  lw=0.8, ls=":")
    ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "AGC Adaptive Gain K_wind", "Timestep", "K_wind")

    # 2,2 — Altitude histogram (riotu GPS real altitude)
    ax = fig.add_subplot(gs[2, 2])
    if riotu_rows:
        alts = [r["altitude"] for r in riotu_rows]
        ax.hist(alts, bins=50, color=GREEN, alpha=0.7, density=True)
        ax.axvline(600, color=RED, lw=1, ls="--", label="600m deploy trigger")
        ax.legend(fontsize=7, facecolor="#161B22", labelcolor=TEXT)
    ax_style(ax, "riotu-GPS: Real Altitude Distribution",
             "Altitude (m)", "Density")

    # ── Row 3: Result summary table ──────────────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    all_results = sead_results + riotu_results
    if all_results:
        col_labels = ["Log", "Dataset", "Rows", "Servo", "delta_a", "No NaN", "Sign", "MeanErr", "RESULT"]
        table_data = []
        for r in all_results:
            ds = "UAV-SEAD" if "UAV-SEAD" in r.get("dataset","") or r["log"].endswith(".ulg") else "riotu-GPS"
            table_data.append([
                r["log"][:22], ds,
                str(r["rows"]),
                r["servo"], r["delta_a"], r["no_nan"], r["sign"],
                f"{r['mean_err']:.1f}°",
                r["overall"],
            ])
        col_colors = [[NAVY]*9]
        cell_colors = []
        for row in table_data:
            rc = []
            for i, v in enumerate(row):
                if v == "PASS":   rc.append("#0d3320")
                elif v == "FAIL": rc.append("#3d0d0d")
                else:             rc.append("#161B22")
            cell_colors.append(rc)

        tbl = ax.table(cellText=table_data, colLabels=col_labels,
                       loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        for (r, c), cell in tbl.get_celld().items():
            if r == 0:
                cell.set_facecolor(NAVY)
                cell.get_text().set_color("white")
                cell.get_text().set_fontweight("bold")
            else:
                cell.set_facecolor(cell_colors[r - 1][c])
                cell.get_text().set_color(TEXT)
            cell.set_edgecolor(GREY)
    ax.set_title("Verification Summary — All Logs", color=TEXT,
                 fontsize=11, fontweight="bold", pad=10)

    # Title
    n_sead  = len(sead_rows)
    n_riotu = len(riotu_rows)
    fig.suptitle(
        f"GARUD AGC Actuation Test — Dual Dataset\n"
        f"UAV-SEAD: {n_sead:,} timesteps  |  riotu-GPS: {n_riotu:,} timesteps  |  "
        f"Total: {n_sead + n_riotu:,} timesteps",
        color=TEXT, fontsize=13, fontweight="bold", y=0.97
    )

    out = RESULTS / "combined_multi_dataset.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [OK] {out.name}  ({out.stat().st_size // 1024} KB)")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ulg",  type=int, default=6,
                        help="Max ULog files to test from UAV-SEAD")
    parser.add_argument("--max-csv",  type=int, default=7,
                        help="Max CSV files to test from riotu-GPS")
    parser.add_argument("--target-ned-x", type=float, default=100.0)
    parser.add_argument("--target-ned-y", type=float, default=50.0)
    args = parser.parse_args()

    print("=" * 65)
    print("  GARUD Actuation Test — DUAL DATASET")
    print("  Dataset A: UAV-SEAD  (HuggingFace, PX4 ULog, binary)")
    print("  Dataset B: riotu-GPS (HuggingFace, outdoor GPS, CSV)")
    print("=" * 65)

    all_rows:    List[dict] = []
    sead_results: List[dict] = []
    riotu_results: List[dict] = []

    # ── DATASET A: UAV-SEAD ULog files ──────────────────────────────────────
    if _ULOG:
        print(f"\n[A] UAV-SEAD  — sampling from multiple dates ...")
        ulg_collected = 0
        for (date, known_files) in SEAD_DATES:
            if ulg_collected >= args.max_ulg:
                break
            files = known_files or discover_sead_files(date, max_files=2)
            if not files:
                print(f"  [SKIP] {date}: no files found via API")
                continue
            for fname in files:
                if ulg_collected >= args.max_ulg:
                    break
                date_dir = DATA_DIR / "sead" / date
                date_dir.mkdir(parents=True, exist_ok=True)
                dest = date_dir / fname
                url  = f"{HF_BASE_SEAD}/ulg_files/{date}/{fname}"
                if not download(url, dest, f"{date}/{fname}"):
                    continue
                rows = parse_ulg(dest, args.target_ned_x, args.target_ned_y)
                if not rows:
                    print(f"  [SKIP] {fname}: no parseable data")
                    continue
                result = verify(rows, fname)
                result["dataset"] = "UAV-SEAD"
                sead_results.append(result)
                all_rows.extend(rows)
                ulg_collected += 1
                sym = "[OK]" if result["overall"] == "PASS" else "[!!]"
                print(f"  {sym} {date}/{fname:<25} {len(rows):>5} rows  "
                      f"mean_err={result['mean_err']:.1f}deg  [{result['overall']}]")
    else:
        print("\n[A] UAV-SEAD SKIPPED — install pyulog:  pip install pyulog")

    # ── DATASET B: riotu-lab GPS CSV files ───────────────────────────────────
    print(f"\n[B] riotu-GPS — {min(args.max_csv, 7)} outdoor GPS trajectory CSVs ...")
    riotu_dir = DATA_DIR / "riotu"
    riotu_dir.mkdir(parents=True, exist_ok=True)
    for fname in RIOTU_FILES[:args.max_csv]:
        url  = f"{HF_BASE_RIOTU}/{fname}"
        dest = riotu_dir / fname
        if not download(url, dest, fname):
            continue
        rows = parse_riotu_csv(dest)
        if not rows:
            print(f"  [SKIP] {fname}: no parseable rows")
            continue
        result = verify(rows, fname)
        result["dataset"] = "riotu-GPS"
        riotu_results.append(result)
        all_rows.extend(rows)
        sym = "[OK]" if result["overall"] == "PASS" else "[!!]"
        print(f"  {sym} {fname:<35} {len(rows):>6} rows  "
              f"mean_err={result['mean_err']:.1f}deg  [{result['overall']}]")

    # ── Summary table ─────────────────────────────────────────────────────────
    all_results = sead_results + riotu_results
    if not all_results:
        print("\n[ERROR] No data parsed from either dataset.")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  GARUD Multi-Dataset Verification Report")
    print(f"  Total timesteps : {len(all_rows):,}")
    print(f"  UAV-SEAD logs   : {len(sead_results)}")
    print(f"  riotu-GPS files : {len(riotu_results)}")
    print(f"{'='*65}")
    hdr = f"  {'Log':<30} {'Rows':>6}  Srv   dA    NaN  Sign  MeanErr"
    print(hdr)
    print("  " + "-" * 63)
    for r in all_results:
        print(f"  {r['log']:<30} {r['rows']:>6}  "
              f"{r['servo']:<5} {r['delta_a']:<5} {r['no_nan']:<5} "
              f"{r['sign']:<5} {r['mean_err']:.1f}deg")
    print("  " + "-" * 63)
    all_pass = all(r["overall"] == "PASS" for r in all_results)
    print(f"\n  OVERALL RESULT: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f"\n[PLOT] Generating combined dashboard ...")
    make_plots(all_rows, sead_results, riotu_results)

    print(f"\n  Output files in: {RESULTS}/")
    print("  Done.")


if __name__ == "__main__":
    main()
