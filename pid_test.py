#!/usr/bin/env python3

import os
import sys
import time

# -----------------------------------------------------------------------------
# Resolve WEBOTS_HOME (same logic as test_cruise.py)
# -----------------------------------------------------------------------------

if "WEBOTS_HOME" in os.environ:
    so_path = os.path.join(
        os.environ["WEBOTS_HOME"],
        "lib",
        "controller",
        "libController.so",
    )
    if not os.path.exists(so_path):
        del os.environ["WEBOTS_HOME"]

if "WEBOTS_HOME" not in os.environ:
    for path in [
        "/opt/ros/humble",
        "/usr/local/webots",
        "/usr/share/webots",
        "/opt/webots",
    ]:
        so_path = os.path.join(
            path,
            "lib",
            "controller",
            "libController.so",
        )
        if os.path.exists(so_path):
            os.environ["WEBOTS_HOME"] = path
            break

if "WEBOTS_HOME" in os.environ:
    python_paths = [
        os.path.join(
            os.environ["WEBOTS_HOME"],
            "lib",
            "controller",
            "python",
        ),
        os.path.join(
            os.environ["WEBOTS_HOME"],
            "local",
            "lib",
            "python3.10",
            "dist-packages",
        ),
    ]

    for p in python_paths:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)


def is_wsl():
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


def get_wsl_host_ip():
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("nameserver"):
                    return line.split()[1]
    except Exception:
        pass

    return "127.0.0.1"


if is_wsl() and "WEBOTS_CONTROLLER_URL" not in os.environ:
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--robot-name", default="vehicle")
    parser.add_argument("--port", default="1234")
    parser.add_argument("--ip", default=None)

    args, _ = parser.parse_known_args()

    host_ip = args.ip if args.ip else get_wsl_host_ip()

    os.environ["WEBOTS_CONTROLLER_URL"] = (
        f"tcp://{host_ip}:{args.port}/{args.robot_name}"
    )

    print("Connecting to:", os.environ["WEBOTS_CONTROLLER_URL"])


try:
    from vehicle import Driver
except ImportError as e:
    print(e)
    sys.exit(1)


# -----------------------------------------------------------------------------
# PID Parameters
# -----------------------------------------------------------------------------

TARGET_SPEED_KMH = 60
TARGET_SPEED = TARGET_SPEED_KMH / 3.6

KP = 0.4
KI = 0.05
KD = 0.1

ACCEL_DEADBAND = 0.1
BRAKE_DEADBAND = -1.0

integral = 0.0
prev_error = 0.0


def clamp(x, low, high):
    return max(low, min(high, x))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

driver = Driver()

TIME_STEP = int(driver.getBasicTimeStep())
DT = TIME_STEP / 1000.0

print(f"Connected to Webots. TIME_STEP={TIME_STEP} ms")

# Initialize GPS
gps = driver.getDevice("gps")
gps.enable(TIME_STEP)

gps_speed = 0.0

def compute_gps_speed():
    global gps_speed
    speed_ms = gps.getSpeed()
    gps_speed = speed_ms * 3.6  # km/h


# Configure initial vehicle settings
driver.setGear(1)
driver.setCruisingSpeed(0.0)  # Cruising speed set to 0 to enable manual throttle control
driver.setThrottle(0.0)
driver.setBrakeIntensity(0.0)
driver.setSteeringAngle(0.0)

print(f"Target Speed = {TARGET_SPEED:.2f} m/s ({TARGET_SPEED*3.6:.2f} km/h)")

steps = 0
while driver.step() != -1:
    compute_gps_speed()

    # GPS speed is in km/h
    current_speed = gps_speed / 3.6

    error = TARGET_SPEED - current_speed

    # PID
    integral += error * DT
    integral = clamp(integral, -1.0, 1.0)

    derivative = (error - prev_error) / DT
    prev_error = error

    raw_output = (
        KP * error
        + KI * integral
        + KD * derivative
    )

    throttle = 0.0
    brake = 0.0

    # only apply throttle or brake if error is large enough
    if error > ACCEL_DEADBAND:
        throttle = clamp(raw_output, 0.0, 1.0)

    elif error < BRAKE_DEADBAND:
        brake = clamp(-raw_output, 0.0, 1.0)

    driver.setThrottle(throttle)
    driver.setBrakeIntensity(brake)
    driver.setSteeringAngle(0.0)

    print(
        f"Speed km/h: {gps_speed:6.2f} km/h | "
        f"Speed m/s: {current_speed:6.2f} m/s | "
        f"Throttle: {throttle:.3f} | "
        f"Brake: {brake:.3f} | "
        f"Error: {error:+.3f} m/s | "
        f"Output: {raw_output:+.3f} | "
        f"Time: {steps * DT:.1f} s"
    )

    if steps * DT > 6:
        TARGET_SPEED = 20 / 3.6

    steps += 1
    time.sleep(DT)