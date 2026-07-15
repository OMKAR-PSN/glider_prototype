# Autonomous Parafoil Mission Control (CAN-7U-SAT)

This repository implements a complete Guidance, Navigation, and Control (GNC) stack for an autonomous single-skin paraglider, designed to be deployed from a sounding rocket payload.

The system uses a **Reinforcement Learning (RL) primary flight controller** (trained with Soft Actor-Critic) and a fallback PID controller. It is designed to run on a Raspberry Pi 4.

## Quick Start

### 1. Software-In-The-Loop (SITL) with Live Dashboard
To run the flight computer in pure simulation mode (SITL):
```bash
cd System001/glider_gnc
python flight_computer.py --sitl
```
This runs the full flight loop (20Hz) against a 4-DOF mathematical physics and environment model, rendering a live Terminal User Interface (TUI) dashboard using the `rich` library. This allows you to test the exact same control logic and RL ONNX model that will fly on the hardware.

### 2. Hardware Deployment
For a full guide on deploying this to the physical Raspberry Pi, including wiring references, servo calibration, and sim-to-real differences, please read:
👉 **[`HARDWARE_GUIDE.md`](HARDWARE_GUIDE.md)**

To run on real hardware (once sensors are wired):
```bash
python flight_computer.py
```

Dependencies for real hardware (Raspberry Pi OS):
- `onnxruntime` (for the RL model)
- `rich` (for the terminal dashboard)
- `stable-baselines3`, `numpy`, `pyyaml`
- `adafruit-circuitpython-servokit` (PCA9685 PWM)
- `smbus2` (I2C sensors)
- `pyserial`, `pynmea2` (GPS and XBee)

### 3. Reinforcement Learning Training
The primary flight controller is a 16-Dimension observation space SAC model.
To train a new RL guidance model from scratch:
```bash
python training/train_sac.py
```
To export your trained SAC model to ONNX for lightweight inference on the Raspberry Pi flight computer:
```bash
python training/export_onnx.py
```

## Architecture Overview
- **`flight_computer.py`**: The main 20Hz control loop.
- **`config/gains.yaml`**: The single source of truth for PID gains, RL model paths, and timeout configurations.
- **`models/`**: Contains the active ONNX model (`sac_policy_6500000_16D.onnx`).
- **`estimation/`**: Madgwick IMU filter, Barometric EKF altitude filter, and RLS Wind Estimator.
- **`hw_interface/`**: Hardware abstraction layer allowing hot-swapping between SITL simulation and physical GPIO/I2C.
- **`state_machine/`**: Handles the transition from BOOST -> DROGUE_DESCENT -> GUIDED_DESCENT.

Detailed architectural state is documented in [`PROJECT_STATE.md`](PROJECT_STATE.md).
