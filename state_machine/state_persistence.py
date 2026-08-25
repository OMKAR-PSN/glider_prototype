"""
state_machine/state_persistence.py
====================================
Payload reset recovery mechanism for GARUD GNC.

Concept ported from payload flight computer (Arduino/Teensy) state logging,
extended with actual boot-time recovery so the Pi can resume mid-flight
after a crash or power glitch.

How it works
------------
1. Every STATE_WRITE_INTERVAL_S seconds, `write_state()` saves a snapshot
   of critical flight variables to `.state` file on disk.
2. On hard state transitions (drogue fired, etc.), `write_state()` is called
   immediately regardless of interval.
3. At boot, `load_state()` reads the file. If the timestamp is within
   MAX_STATE_AGE_S seconds, the system resumes from the saved state instead
   of starting from BOOT.

Safety rules (non-negotiable)
------------------------------
- `drogue_fired=True` in .state → drogue channel is LOCKED OUT. Cannot re-fire.
- `drogue_fired=False` in .state → normal operation.
- If .state timestamp > MAX_STATE_AGE_S old → treat as cold start (stale file).
- If .state file is corrupt/missing → treat as cold start.

State file format (JSON, one file, atomic overwrite)
-----------------------------------------------------
{
    "schema_version": 1,
    "timestamp_utc":  "2026-08-25T14:30:00Z",
    "timestamp_mono": 12345.67,          # monotonic seconds since Pi boot
    "flight_state":   "GUIDED_DESCENT",
    "ground_altitude_m": 560.3,          # MSL altitude of launch pad (m)
    "drogue_fired":   true,              # SAFETY LOCK — never re-fire if true
    "last_altitude_m": 320.1,            # last known AGL altitude
    "last_velocity_ms": -3.2,            # last known vertical velocity
    "target_lat":     18.5204,           # mission target
    "target_lon":     73.8567,
    "rl_active":      true               # whether RL was active at crash
}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STATE_FILE_PATH       = Path("flight.state")   # written to cwd (glider_gnc/)
STATE_WRITE_INTERVAL_S = 5.0                   # write every 5 seconds in flight
MAX_STATE_AGE_S        = 120.0                 # ignore .state files older than 2 min
SCHEMA_VERSION         = 1


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    """All variables needed to resume flight after a reboot."""

    # Core state
    flight_state:      str    # "BOOT" | "GUIDED_DESCENT" | "FLARE" | "LANDED"
    ground_altitude_m: float  # MSL altitude of launch pad (for EKF reference)

    # Safety locks — NEVER allow re-fire if True
    drogue_fired: bool

    # Last known kinematics
    last_altitude_m:  float   # AGL altitude at time of crash
    last_velocity_ms: float   # vertical velocity (m/s) at time of crash

    # Mission target
    target_lat: float
    target_lon: float

    # Controller state
    rl_active: bool

    # Timestamps (filled automatically by write_state)
    timestamp_utc:  str   = ""
    timestamp_mono: float = 0.0
    schema_version: int   = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_state(snapshot: StateSnapshot, path: Path = STATE_FILE_PATH) -> None:
    """
    Atomically write snapshot to .state file.

    Uses write-to-temp + rename pattern to prevent partial writes
    (a power cut during write leaves the old file intact).
    """
    snapshot.timestamp_utc  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    snapshot.timestamp_mono = time.monotonic()

    tmp_path = path.with_suffix(".state.tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(snapshot), f, indent=2)
        os.replace(tmp_path, path)   # atomic on Linux/POSIX
        logger.debug("State written: %s  drogue=%s  alt=%.1f m",
                     snapshot.flight_state, snapshot.drogue_fired,
                     snapshot.last_altitude_m)
    except Exception as e:
        logger.warning("Failed to write .state file: %s", e)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_state(path: Path = STATE_FILE_PATH) -> Optional[StateSnapshot]:
    """
    Load .state file and return a StateSnapshot if valid and fresh.

    Returns None if:
      - File does not exist         → cold start
      - File is corrupt / bad JSON  → cold start
      - Schema version mismatch     → cold start
      - Timestamp older than MAX_STATE_AGE_S → stale, cold start
    """
    if not path.exists():
        logger.info("No .state file found — cold start.")
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(".state file corrupt (%s) — cold start.", e)
        _archive_bad_state(path)
        return None

    # Schema check
    if data.get("schema_version") != SCHEMA_VERSION:
        logger.warning(".state schema version mismatch (got %s, want %s) — cold start.",
                       data.get("schema_version"), SCHEMA_VERSION)
        _archive_bad_state(path)
        return None

    # Age check — use wall-clock difference vs stored UTC timestamp
    try:
        saved_at = datetime.strptime(
            data["timestamp_utc"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        age_s = (datetime.now(timezone.utc) - saved_at).total_seconds()
    except Exception:
        logger.warning(".state has bad timestamp — cold start.")
        _archive_bad_state(path)
        return None

    if age_s > MAX_STATE_AGE_S:
        logger.info(".state is %.0f s old (max %.0f s) — stale, cold start.",
                    age_s, MAX_STATE_AGE_S)
        _archive_bad_state(path)
        return None

    # Build snapshot
    try:
        snapshot = StateSnapshot(
            flight_state      = str(data["flight_state"]),
            ground_altitude_m = float(data["ground_altitude_m"]),
            drogue_fired      = bool(data["drogue_fired"]),
            last_altitude_m   = float(data["last_altitude_m"]),
            last_velocity_ms  = float(data["last_velocity_ms"]),
            target_lat        = float(data["target_lat"]),
            target_lon        = float(data["target_lon"]),
            rl_active         = bool(data["rl_active"]),
            timestamp_utc     = data["timestamp_utc"],
            timestamp_mono    = float(data["timestamp_mono"]),
            schema_version    = int(data["schema_version"]),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(".state missing field (%s) — cold start.", e)
        _archive_bad_state(path)
        return None

    logger.info(
        "Recovered .state: state=%s  drogue=%s  alt=%.1f m  age=%.1f s",
        snapshot.flight_state, snapshot.drogue_fired,
        snapshot.last_altitude_m, age_s,
    )
    return snapshot


# ---------------------------------------------------------------------------
# Delete / Archive
# ---------------------------------------------------------------------------

def delete_state(path: Path = STATE_FILE_PATH) -> None:
    """Remove the .state file on clean landing (no stale recovery next boot)."""
    try:
        path.unlink(missing_ok=True)
        logger.info(".state file deleted (clean landing).")
    except Exception as e:
        logger.warning("Could not delete .state file: %s", e)


def _archive_bad_state(path: Path) -> None:
    """Rename bad .state file to .state.bad for post-flight inspection."""
    bad_path = path.with_suffix(".state.bad")
    try:
        os.replace(path, bad_path)
        logger.info("Bad .state archived to %s", bad_path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Parser / viewer (run standalone to inspect a .state file)
# ---------------------------------------------------------------------------

def print_state_file(path: Path = STATE_FILE_PATH) -> None:
    """Pretty-print the .state file for ground crew inspection."""
    snapshot = load_state(path)
    if snapshot is None:
        print(f"No valid .state file at {path}")
        return

    print("=" * 45)
    print("  GARUD FLIGHT STATE SNAPSHOT")
    print("=" * 45)
    print(f"  Saved at       : {snapshot.timestamp_utc}")
    print(f"  Flight state   : {snapshot.flight_state}")
    print(f"  Ground alt MSL : {snapshot.ground_altitude_m:.1f} m")
    print(f"  Last alt AGL   : {snapshot.last_altitude_m:.1f} m")
    print(f"  Last velocity  : {snapshot.last_velocity_ms:.2f} m/s")
    print(f"  Drogue fired   : {'YES — LOCKED OUT' if snapshot.drogue_fired else 'No'}")
    print(f"  Target         : {snapshot.target_lat:.6f}, {snapshot.target_lon:.6f}")
    print(f"  RL was active  : {snapshot.rl_active}")
    print("=" * 45)


if __name__ == "__main__":
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else STATE_FILE_PATH
    print_state_file(p)
