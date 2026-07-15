import math

def check_scenario_reachability(altitude, glide_ratio, airspeed, distance_to_target, wind_speed, wind_direction, bearing_to_target):
    """
    Validates whether a target is physically reachable under the given conditions.
    Returns: (is_reachable: bool, reason: str)
    """
    max_still_air_range = altitude * glide_ratio
    
    # Calculate the angle difference between wind direction and bearing to target.
    # We use cos() so the sign of the difference doesn't matter.
    angle_diff = wind_direction - bearing_to_target
    tailwind_component = wind_speed * math.cos(angle_diff)
    
    effective_groundspeed_along_path = airspeed + tailwind_component
    
    # Short-circuit on negative/zero groundspeed to prevent div-by-zero or negative ranges
    if effective_groundspeed_along_path <= 0.0:
        return False, "headwind_exceeds_airspeed"
        
    safety_margin = 1.3
    range_ratio = effective_groundspeed_along_path / airspeed
    max_effective_range = max_still_air_range * range_ratio
    
    if max_effective_range >= (distance_to_target * safety_margin):
        return True, "reachable"
    else:
        return False, "too_far_for_glide_ratio"
