#!/usr/bin/env python3
import os
import sys
import time
import math

# Resolve WEBOTS_HOME
# If WEBOTS_HOME is already set, check if it is missing the Linux shared library (e.g. Windows installation)
if 'WEBOTS_HOME' in os.environ:
    so_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'libController.so')
    if not os.path.exists(so_path):
        del os.environ['WEBOTS_HOME']

if 'WEBOTS_HOME' not in os.environ:
    for path in ['/opt/ros/humble', '/usr/local/webots', '/usr/share/webots', '/opt/webots']:
        so_path = os.path.join(path, 'lib', 'controller', 'libController.so')
        if os.path.exists(so_path):
            os.environ['WEBOTS_HOME'] = path
            break

if 'WEBOTS_HOME' in os.environ:
    paths_to_append = [
        os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python'),
        os.path.join(os.environ['WEBOTS_HOME'], 'local', 'lib', 'python3.10', 'dist-packages')
    ]
    for p in paths_to_append:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)

def is_wsl():
    try:
        with open('/proc/version', 'r') as f:
            return 'microsoft' in f.read().lower()
    except Exception:
        return False

def get_wsl_host_ip():
    try:
        with open('/etc/resolv.conf', 'r') as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                tokens = line.split()
                if tokens and tokens[0] == 'nameserver':
                    return tokens[1]
    except Exception:
        pass
    return '127.0.0.1'

if is_wsl() and 'WEBOTS_CONTROLLER_URL' not in os.environ:
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--robot-name', type=str, default='vehicle')
    parser.add_argument('--port', type=str, default='1234')
    parser.add_argument('--ip', type=str, default=None)
    args, _ = parser.parse_known_args()
    
    host_ip = args.ip if args.ip else get_wsl_host_ip()
    if args.robot_name:
        url = f"tcp://{host_ip}:{args.port}/{args.robot_name}"
    else:
        url = f"tcp://{host_ip}:{args.port}/"
        
    os.environ['WEBOTS_CONTROLLER_URL'] = url
    print(f"WSL environment detected. Connecting to Windows Webots via: {url}")

try:
    from vehicle import Driver
except ImportError as e:
    print(f"Error importing Webots Driver API: {e}", file=sys.stderr)
    sys.exit(1)

def main():
    print("Initializing Driver...")
    try:
        driver = Driver()
    except Exception as e:
        print(f"Failed to initialize Driver: {e}", file=sys.stderr)
        sys.exit(1)

    timestep = int(driver.getBasicTimeStep())
    print(f"Connected to Webots. Timestep: {timestep}ms")

    # Initialize GPS
    gps = driver.getDevice("gps")
    if gps is not None:
        gps.enable(timestep)
    else:
        print("GPS device not found on the vehicle!", file=sys.stderr)

    # Set target cruising speed (km/h)
    target_speed_kmh = 50
    print(f"Setting target cruising speed to {target_speed_kmh} km/h")
    driver.setCruisingSpeed(target_speed_kmh)
    
    # Keep wheels straight
    driver.setSteeringAngle(0.0)

    try:
        while driver.step() != -1:
            current_speed = driver.getCurrentSpeed()
            sim_time = driver.getTime()
            if gps is not None:
                gps_speed_val = gps.getSpeed() * 3.6
                gps_coords = gps.getValues()
                gps_str = f" | GPS Speed: {gps_speed_val:.3f} km/h | GPS Coords: {[round(c, 4) if not math.isnan(c) else c for c in gps_coords]}"
            else:
                gps_str = ""
            print(f"[{sim_time:.3f}s] Target: {target_speed_kmh} km/h | Current Speed: {current_speed:.3f} km/h{gps_str}")
            time.sleep(timestep / 1000.0)
    except KeyboardInterrupt:
        print("Exiting...")
        driver.setCruisingSpeed(0.0)
        driver.step()

if __name__ == '__main__':
    main()
