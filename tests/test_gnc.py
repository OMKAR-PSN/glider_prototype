import math
import pytest
from control.coordinated_turn import bank_angle_to_turn_rate, turn_rate_to_bank_angle
from control.roll_pid import RollPID
from guidance.altitude_budget import AltitudeBudget
from mixing.elevon_mixer import ElevonMixer

def test_coordinated_turn_conversions():
    airspeed = 15.0
    gravity = 9.81
    
    # 20 deg bank
    bank = math.radians(20)
    turn_rate = bank_angle_to_turn_rate(bank, airspeed)
    
    # Check forward
    expected_rate = (gravity * math.tan(bank)) / airspeed
    assert pytest.approx(turn_rate, rel=1e-4) == expected_rate
    
    # Check inverse
    bank_inv = turn_rate_to_bank_angle(turn_rate, airspeed)
    assert pytest.approx(bank_inv, rel=1e-4) == bank

def test_roll_pid():
    pid = RollPID(kp=1.5, ki=0.1, kd=0.05, integral_limit=10.0)
    
    cmd = math.radians(10)
    meas = math.radians(0)
    rate = 0.0
    dt = 0.05
    
    effort = pid.step(cmd, meas, rate, dt)
    assert effort > 0 # Should command positive effort

def test_altitude_budget():
    budget = AltitudeBudget(glide_ratio=5.0, s_turn_excess_threshold=15.0, s_turn_bank_angle_rad=math.radians(20))
    
    # distance 500m, needs 100m alt to glide.
    # We are at 150m, target is 0m. Alt excess is 50m.
    # threshold is 15. So we should S-turn.
    turn_cmd = budget.compute(150.0, 0.0, 500.0, 0.0)
    assert turn_cmd != 0.0
    assert abs(turn_cmd) == math.radians(20)

    # If we are at 105m, excess is 5m. < 15. Normal guidance.
    turn_cmd2 = budget.compute(105.0, 0.0, 500.0, 0.0)
    assert turn_cmd2 == 0.0

def test_elevon_mixer():
    mixer = ElevonMixer(60.0, 120.0, 90.0, 0.0, 0.0)
    
    # Pure pitch up (positive pitch cmd)
    left, right = mixer.mix(roll_cmd=0.0, pitch_cmd=10.0)
    assert left == 100.0
    assert right == 100.0
    
    # Pure roll right (positive roll cmd) -> left elevon down (more pitch/lift), right elevon up (less)
    left, right = mixer.mix(roll_cmd=10.0, pitch_cmd=0.0)
    assert left == 80.0
    assert right == 100.0
    
    # Saturation
    left, right = mixer.mix(roll_cmd=50.0, pitch_cmd=50.0)
    assert left >= 60.0 and left <= 120.0
    assert right >= 60.0 and right <= 120.0
