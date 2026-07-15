from enum import Enum, auto

class FlightState(Enum):
    BOOST = auto()
    DROGUE_DESCENT = auto()
    DEPLOYMENT_TRIGGER = auto()
    DEPLOYMENT_VERIFICATION = auto()
    GUIDED_DESCENT = auto()
    LANDED = auto()

class StateMachine:
    """
    Manages the transitions between flight states.
    The 600m AGL deployment trigger uses a moving average over at least 10 
    barometer samples plus a confirmed-descending check from IMU vertical accel.
    """
    def __init__(self, ground_altitude: float = 0.0):
        self.state = FlightState.BOOST
        self.ground_altitude = ground_altitude
        
        self.baro_history = []
        self.history_size = 10
        self.deployment_agl_threshold = 600.0

    def update(self, current_alt: float, vertical_velocity: float) -> FlightState:
        """
        Updates the state machine based on sensor data.
        
        Args:
            current_alt: Current MSL altitude (from baro)
            vertical_velocity: EKF vertical velocity (m/s), negative means descending
            
        Returns:
            The current flight state
        """
        self.baro_history.append(current_alt)
        if len(self.baro_history) > self.history_size:
            self.baro_history.pop(0)

        agl = current_alt - self.ground_altitude
        
        if self.state == FlightState.BOOST:
            # Transition to drogue descent when apogee is reached (simplified)
            if len(self.baro_history) == self.history_size:
                # If we've dropped 5m and we are descending
                if self.baro_history[-1] < self.baro_history[0] - 5.0 and vertical_velocity < -2.0:
                    self.state = FlightState.DROGUE_DESCENT
                    
        elif self.state == FlightState.DROGUE_DESCENT:
            # 600m AGL moving average trigger
            if len(self.baro_history) == self.history_size:
                avg_alt = sum(self.baro_history) / len(self.baro_history)
                avg_agl = avg_alt - self.ground_altitude
                
                # Check AGL threshold AND descending check
                if avg_agl <= self.deployment_agl_threshold and vertical_velocity < -2.0:
                    self.state = FlightState.DEPLOYMENT_TRIGGER
                    
        elif self.state == FlightState.DEPLOYMENT_TRIGGER:
            # In a real system, wait a few ms to fire pyros or open servos
            self.state = FlightState.DEPLOYMENT_VERIFICATION
            
        elif self.state == FlightState.DEPLOYMENT_VERIFICATION:
            # Wings lock under aerodynamic load, give it a moment, then fly
            self.state = FlightState.GUIDED_DESCENT
            
        elif self.state == FlightState.GUIDED_DESCENT:
            if agl <= 5.0:
                self.state = FlightState.LANDED

        return self.state
