"""
tests/test_glider_servos.py
===========================
Standalone hardware test for all servos on the GARUD HAT via PCA9685.

Hardware (confirmed from GARUD HAT schematic Rev 2):
  PCA9685 servo driver : I2C bus, SDA=GPIO2, SCL=GPIO3
  OE_Servo             : GPIO4 (output enable, active LOW)
  PCA9685 I2C address  : 0x40 (A0-A5 all tied to GND)

  Channel 0  MG995R  Left glider brake
  Channel 1  MG995R  Right glider brake
  Channel 2  MG995R  Drogue release servo (one-shot: 0deg -> 120deg at 600m AGL)
  Channel 3  SG90    Gimbal roll stabilisation
  Channel 4  SG90    Gimbal pitch stabilisation

MG995R PWM timing  : 1000us (60deg) | 1500us (90deg neutral) | 2000us (120deg)
SG90   PWM timing  : 500us  (0deg)  | 1500us (90deg centre)  | 2400us (180deg)

Run on Raspberry Pi 4:
  python tests/test_glider_servos.py

Install dependencies first (run once on Pi):
  pip install adafruit-circuitpython-pca9685

Keyboard controls:
  n   both glider servos NEUTRAL (90 deg)
  l   LEFT brake  -- ch0=120 deg, ch1=90 deg
  r   RIGHT brake -- ch0=90 deg,  ch1=120 deg
  b   BOTH brakes -- ch0=ch1=120 deg (flare / landing)
  d   DROGUE release -- ch2: 0 deg -> 120 deg (one-way, irreversible in flight)
  [   left servo  -5 deg
  ]   left servo  +5 deg
  ,   right servo -5 deg
  .   right servo +5 deg
  g   GIMBAL centre (ch3=ch4=90 deg)
  s   SWEEP test   -- slow sweep on glider servos only
  f   FULL SEQUENCE test -- automated flight sequence
  q   QUIT and return all servos to neutral

Author  : Team Anantam (GARUD, IN-SPACe 2026-002)
Version : 1.0.0
"""

import sys
import time

# ---------------------------------------------------------------------------
# Pin / channel configuration (read from gains.yaml in flight; hardcoded here
# for standalone testing only)
# ---------------------------------------------------------------------------

PCA9685_ADDRESS = 0x40      # A0-A5 tied to GND on GARUD HAT

# PCA9685 channel assignments
CH_LEFT   = 0   # MG995R -- left glider brake
CH_RIGHT  = 1   # MG995R -- right glider brake
CH_DROGUE = 2   # MG995R -- drogue release
CH_GIMBAL_ROLL  = 3   # SG90 -- gimbal roll
CH_GIMBAL_PITCH = 4   # SG90 -- gimbal pitch

SERVO_FREQ_HZ = 50      # 50 Hz standard servo PWM

# MG995R limits (degrees and matching pulse widths in microseconds)
MG995R_MIN_DEG   = 60.0
MG995R_MID_DEG   = 90.0
MG995R_MAX_DEG   = 120.0
MG995R_MIN_US    = 1000
MG995R_MID_US    = 1500
MG995R_MAX_US    = 2000

# SG90 limits
SG90_MIN_DEG = 0.0
SG90_MID_DEG = 90.0
SG90_MAX_DEG = 180.0
SG90_MIN_US  = 500
SG90_MID_US  = 1500
SG90_MAX_US  = 2400

# Drogue release positions
DROGUE_LOCKED_DEG   = 60.0    # holds drogue closed
DROGUE_RELEASED_DEG = 120.0   # one-shot release

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def deg_to_duty(deg, min_deg, max_deg, min_us, max_us):
    """Map degrees to PCA9685 duty cycle (0.0-1.0)."""
    deg = max(min_deg, min(max_deg, deg))
    t = (deg - min_deg) / (max_deg - min_deg)
    pulse_us = min_us + t * (max_us - min_us)
    period_us = 1_000_000.0 / SERVO_FREQ_HZ   # 20000 us at 50 Hz
    return pulse_us / period_us


def set_mg995r(pca, channel, deg):
    """Set an MG995R servo channel to the requested angle (60-120 deg)."""
    deg = max(MG995R_MIN_DEG, min(MG995R_MAX_DEG, deg))
    duty = deg_to_duty(deg, MG995R_MIN_DEG, MG995R_MAX_DEG,
                       MG995R_MIN_US, MG995R_MAX_US)
    pca.channels[channel].duty_cycle = int(duty * 0xFFFF)
    print(f"    CH{channel} (MG995R) -> {deg:.1f} deg  [{int(duty*100000)/1000:.2f}% duty]")


def set_sg90(pca, channel, deg):
    """Set an SG90 servo channel to the requested angle (0-180 deg)."""
    deg = max(SG90_MIN_DEG, min(SG90_MAX_DEG, deg))
    duty = deg_to_duty(deg, SG90_MIN_DEG, SG90_MAX_DEG,
                       SG90_MIN_US, SG90_MAX_US)
    pca.channels[channel].duty_cycle = int(duty * 0xFFFF)
    print(f"    CH{channel} (SG90)   -> {deg:.1f} deg  [{int(duty*100000)/1000:.2f}% duty]")


def glider_neutral(pca):
    print("\n[NEUTRAL] Glider servos ch0+ch1 -> 90 deg")
    set_mg995r(pca, CH_LEFT,  MG995R_MID_DEG)
    set_mg995r(pca, CH_RIGHT, MG995R_MID_DEG)


def gimbal_centre(pca):
    print("\n[GIMBAL CENTRE] ch3+ch4 -> 90 deg")
    set_sg90(pca, CH_GIMBAL_ROLL,  SG90_MID_DEG)
    set_sg90(pca, CH_GIMBAL_PITCH, SG90_MID_DEG)


def all_neutral(pca):
    """Set every servo to its neutral/safe position."""
    glider_neutral(pca)
    set_mg995r(pca, CH_DROGUE, DROGUE_LOCKED_DEG)
    gimbal_centre(pca)


# ---------------------------------------------------------------------------
# Test routines
# ---------------------------------------------------------------------------

def sweep_test(pca):
    """Slow symmetric sweep on both glider servos simultaneously."""
    print("\n[SWEEP TEST] Glider servos: 60 -> 120 -> 60 deg ...")
    steps = 30
    for i in range(steps + 1):
        deg = MG995R_MIN_DEG + i * (MG995R_MAX_DEG - MG995R_MIN_DEG) / steps
        pca.channels[CH_LEFT].duty_cycle = int(
            deg_to_duty(deg, MG995R_MIN_DEG, MG995R_MAX_DEG,
                        MG995R_MIN_US, MG995R_MAX_US) * 0xFFFF)
        pca.channels[CH_RIGHT].duty_cycle = int(
            deg_to_duty(deg, MG995R_MIN_DEG, MG995R_MAX_DEG,
                        MG995R_MIN_US, MG995R_MAX_US) * 0xFFFF)
        time.sleep(0.05)
    for i in range(steps + 1):
        deg = MG995R_MAX_DEG - i * (MG995R_MAX_DEG - MG995R_MIN_DEG) / steps
        pca.channels[CH_LEFT].duty_cycle = int(
            deg_to_duty(deg, MG995R_MIN_DEG, MG995R_MAX_DEG,
                        MG995R_MIN_US, MG995R_MAX_US) * 0xFFFF)
        pca.channels[CH_RIGHT].duty_cycle = int(
            deg_to_duty(deg, MG995R_MIN_DEG, MG995R_MAX_DEG,
                        MG995R_MIN_US, MG995R_MAX_US) * 0xFFFF)
        time.sleep(0.05)
    glider_neutral(pca)
    print("[SWEEP TEST] Done.")


def gimbal_sweep_test(pca):
    """Sweep gimbal servos to verify range."""
    print("\n[GIMBAL SWEEP] ch3+ch4: 0 -> 180 -> 0 deg ...")
    steps = 30
    for i in range(steps + 1):
        deg = SG90_MIN_DEG + i * (SG90_MAX_DEG - SG90_MIN_DEG) / steps
        pca.channels[CH_GIMBAL_ROLL].duty_cycle = int(
            deg_to_duty(deg, SG90_MIN_DEG, SG90_MAX_DEG,
                        SG90_MIN_US, SG90_MAX_US) * 0xFFFF)
        pca.channels[CH_GIMBAL_PITCH].duty_cycle = int(
            deg_to_duty(deg, SG90_MIN_DEG, SG90_MAX_DEG,
                        SG90_MIN_US, SG90_MAX_US) * 0xFFFF)
        time.sleep(0.06)
    for i in range(steps + 1):
        deg = SG90_MAX_DEG - i * (SG90_MAX_DEG - SG90_MIN_DEG) / steps
        pca.channels[CH_GIMBAL_ROLL].duty_cycle = int(
            deg_to_duty(deg, SG90_MIN_DEG, SG90_MAX_DEG,
                        SG90_MIN_US, SG90_MAX_US) * 0xFFFF)
        pca.channels[CH_GIMBAL_PITCH].duty_cycle = int(
            deg_to_duty(deg, SG90_MIN_DEG, SG90_MAX_DEG,
                        SG90_MIN_US, SG90_MAX_US) * 0xFFFF)
        time.sleep(0.06)
    gimbal_centre(pca)
    print("[GIMBAL SWEEP] Done.")


def full_sequence_test(pca):
    """
    Runs the full automated flight control sequence.
    Simulates what the flight computer will command during descent.
    """
    print("\n" + "="*50)
    print("  FULL SEQUENCE TEST")
    print("  Simulates the complete descent control profile.")
    print("="*50)

    print("\n[1/8] NEUTRAL -- all servos safe position (2 sec)")
    all_neutral(pca)
    time.sleep(2)

    print("\n[2/8] LEFT BRAKE -- ch0=120 deg, ch1=90 deg (2 sec)")
    set_mg995r(pca, CH_LEFT,  MG995R_MAX_DEG)
    set_mg995r(pca, CH_RIGHT, MG995R_MID_DEG)
    time.sleep(2)

    print("\n[3/8] NEUTRAL (1 sec)")
    glider_neutral(pca)
    time.sleep(1)

    print("\n[4/8] RIGHT BRAKE -- ch0=90 deg, ch1=120 deg (2 sec)")
    set_mg995r(pca, CH_LEFT,  MG995R_MID_DEG)
    set_mg995r(pca, CH_RIGHT, MG995R_MAX_DEG)
    time.sleep(2)

    print("\n[5/8] NEUTRAL (1 sec)")
    glider_neutral(pca)
    time.sleep(1)

    print("\n[6/8] GRADUAL LEFT TURN -- 10%% -> 100%% deflection")
    for pct in [0.1, 0.25, 0.5, 0.75, 1.0]:
        da = pct * (MG995R_MAX_DEG - MG995R_MID_DEG)
        set_mg995r(pca, CH_LEFT,  MG995R_MID_DEG + da)
        set_mg995r(pca, CH_RIGHT, MG995R_MID_DEG)
        time.sleep(0.5)
    glider_neutral(pca)
    time.sleep(1)

    print("\n[7/8] SYMMETRIC BRAKE / FLARE -- both ch0+ch1=120 deg (2 sec)")
    set_mg995r(pca, CH_LEFT,  MG995R_MAX_DEG)
    set_mg995r(pca, CH_RIGHT, MG995R_MAX_DEG)
    time.sleep(2)

    print("\n[8/8] Return to NEUTRAL")
    all_neutral(pca)

    print("\n" + "="*50)
    print("  FULL SEQUENCE TEST COMPLETE")
    print("="*50)


def drogue_test(pca):
    """
    Test drogue release servo (ch2 MG995R).
    WARNING: This moves the servo to the RELEASE position.
             Do NOT run this test with the canopy attached.
    """
    print("\n[DROGUE TEST]")
    print("  WARNING: Do NOT run with canopy attached.")
    print("  CH2 will move from LOCKED (60 deg) to RELEASED (120 deg).")
    confirm = input("  Type YES to continue: ")
    if confirm.strip().upper() != "YES":
        print("  Cancelled.")
        return
    print("  CH2 -> LOCKED (60 deg)")
    set_mg995r(pca, CH_DROGUE, DROGUE_LOCKED_DEG)
    time.sleep(1)
    print("  CH2 -> RELEASING ... (120 deg)")
    set_mg995r(pca, CH_DROGUE, DROGUE_RELEASED_DEG)
    time.sleep(1)
    print("  CH2 -> LOCKED (60 deg) -- reset for next test")
    set_mg995r(pca, CH_DROGUE, DROGUE_LOCKED_DEG)
    print("  [DROGUE TEST] Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Import hardware libraries
    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
    except ImportError:
        print("ERROR: Hardware library missing.")
        print("Run:  pip install adafruit-circuitpython-pca9685")
        sys.exit(1)

    # Connect to PCA9685
    print(f"\nConnecting to PCA9685 at I2C 0x{PCA9685_ADDRESS:02X} ...")
    print("  SDA=GPIO2 (Pin 3)  SCL=GPIO3 (Pin 5)  OE=GPIO4 (Pin 7)")
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=PCA9685_ADDRESS)
        pca.frequency = SERVO_FREQ_HZ
        print(f"  PCA9685 OK  (freq={SERVO_FREQ_HZ}Hz)")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Checks:")
        print("  sudo raspi-config -> Interface Options -> I2C -> Enable")
        print("  sudo i2cdetect -y 1  (should show 0x40)")
        sys.exit(1)

    # Safe startup position
    print("\nMoving all servos to safe position ...")
    all_neutral(pca)
    time.sleep(0.5)

    left_pos  = MG995R_MID_DEG
    right_pos = MG995R_MID_DEG

    print("""
============================================================
  GARUD HAT Servo Test  |  Team Anantam  |  IN-SPACe 2026
  PCA9685 I2C 0x40
  CH0=Left(MG995R)  CH1=Right(MG995R)  CH2=Drogue(MG995R)
  CH3=GimbalRoll(SG90)  CH4=GimbalPitch(SG90)
============================================================
  n   Glider NEUTRAL  (ch0+ch1 = 90 deg)
  l   LEFT brake      (ch0=120, ch1=90)
  r   RIGHT brake     (ch0=90,  ch1=120)
  b   BOTH brakes     (ch0=ch1=120  -- flare)
  d   DROGUE release  (ch2: 60->120 deg, with confirmation)
  [   left servo  -5 deg      ]   left servo  +5 deg
  ,   right servo -5 deg      .   right servo +5 deg
  g   Gimbal CENTRE   (ch3+ch4 = 90 deg)
  G   Gimbal SWEEP    (ch3+ch4 sweep 0-180-0 deg)
  s   SWEEP test      (glider servos ch0+ch1)
  f   FULL SEQUENCE test
  q   QUIT  (all servos to neutral, then off)
============================================================
""")

    # Raw keyboard input (Linux/Pi only)
    import termios, tty

    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    try:
        while True:
            key = getch()

            if key == 'q':
                print("\nQuitting -- returning all servos to neutral ...")
                all_neutral(pca)
                time.sleep(0.3)
                break

            elif key == 'n':
                left_pos = right_pos = MG995R_MID_DEG
                glider_neutral(pca)

            elif key == 'l':
                print("\n[LEFT BRAKE]")
                left_pos = MG995R_MAX_DEG
                right_pos = MG995R_MID_DEG
                set_mg995r(pca, CH_LEFT,  left_pos)
                set_mg995r(pca, CH_RIGHT, right_pos)

            elif key == 'r':
                print("\n[RIGHT BRAKE]")
                left_pos = MG995R_MID_DEG
                right_pos = MG995R_MAX_DEG
                set_mg995r(pca, CH_LEFT,  left_pos)
                set_mg995r(pca, CH_RIGHT, right_pos)

            elif key == 'b':
                print("\n[BOTH BRAKES -- FLARE]")
                left_pos = right_pos = MG995R_MAX_DEG
                set_mg995r(pca, CH_LEFT,  left_pos)
                set_mg995r(pca, CH_RIGHT, right_pos)

            elif key == 'd':
                drogue_test(pca)

            elif key == '[':
                left_pos = max(MG995R_MIN_DEG, left_pos - 5.0)
                set_mg995r(pca, CH_LEFT, left_pos)

            elif key == ']':
                left_pos = min(MG995R_MAX_DEG, left_pos + 5.0)
                set_mg995r(pca, CH_LEFT, left_pos)

            elif key == ',':
                right_pos = max(MG995R_MIN_DEG, right_pos - 5.0)
                set_mg995r(pca, CH_RIGHT, right_pos)

            elif key == '.':
                right_pos = min(MG995R_MAX_DEG, right_pos + 5.0)
                set_mg995r(pca, CH_RIGHT, right_pos)

            elif key == 'g':
                gimbal_centre(pca)

            elif key == 'G':
                gimbal_sweep_test(pca)

            elif key == 's':
                sweep_test(pca)
                left_pos = right_pos = MG995R_MID_DEG

            elif key == 'f':
                full_sequence_test(pca)
                left_pos = right_pos = MG995R_MID_DEG

            else:
                print(f"  (unknown key '{key}' -- press q to quit)")

    except KeyboardInterrupt:
        print("\n\nCtrl+C -- returning all servos to neutral ...")
        all_neutral(pca)

    finally:
        all_neutral(pca)
        pca.deinit()
        print("PCA9685 released. Bye.")


if __name__ == '__main__':
    main()
