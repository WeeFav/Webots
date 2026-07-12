#!/usr/bin/env python3
import os
import sys
import math
import pygame
import matplotlib.pyplot as plt

# ==============================================================================
# Webots Environment & Path Setup
# ==============================================================================
# If WEBOTS_HOME is already set, check if it is missing the Linux shared library (e.g. Windows installation)
if 'WEBOTS_HOME' in os.environ:
    so_path = os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'libController.so')
    if not os.path.exists(so_path):
        print(f"Note: Current WEBOTS_HOME ({os.environ['WEBOTS_HOME']}) is missing 'libController.so' (Windows path). Searching for Linux path...", file=sys.stderr)
        del os.environ['WEBOTS_HOME']

# Try to resolve WEBOTS_HOME if not already set in environment
if 'WEBOTS_HOME' not in os.environ:
    # Look for common default Linux installations containing the shared library
    for path in ['/opt/ros/humble', '/usr/local/webots', '/usr/share/webots', '/opt/webots']:
        so_path = os.path.join(path, 'lib', 'controller', 'libController.so')
        if os.path.exists(so_path):
            os.environ['WEBOTS_HOME'] = path
            break

if 'WEBOTS_HOME' in os.environ:
    # Append common python paths for Webots controller API
    paths_to_append = [
        os.path.join(os.environ['WEBOTS_HOME'], 'lib', 'controller', 'python'),
        os.path.join(os.environ['WEBOTS_HOME'], 'local', 'lib', 'python3.10', 'dist-packages')
    ]
    for p in paths_to_append:
        if os.path.exists(p) and p not in sys.path:
            sys.path.append(p)
# ==============================================================================
# WSL to Windows Host TCP Connection Setup
# ==============================================================================
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
    print("Make sure WEBOTS_HOME is correctly set and points to a Webots installation.", file=sys.stderr)
    print("Example: WEBOTS_HOME=/opt/ros/humble python3 webots_manual_control.py", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# UI Palette Constants
# ==============================================================================
COLOR_BG = (15, 23, 42)          # Tailwind slate-900 (Dark Slate)
COLOR_PANEL = (30, 41, 59)       # Tailwind slate-800 (Panel BG)
COLOR_BORDER = (51, 65, 85)      # Tailwind slate-700
COLOR_TEXT_PRIMARY = (248, 250, 252)  # Tailwind slate-50 (Near White)
COLOR_TEXT_MUTED = (148, 163, 184)    # Tailwind slate-400 (Slate Gray)

COLOR_ACCENT_GREEN = (16, 185, 129)   # Tailwind emerald-500 (Throttle / Success)
COLOR_ACCENT_RED = (239, 68, 68)      # Tailwind red-500 (Brake / Warning)
COLOR_ACCENT_BLUE = (59, 130, 246)     # Tailwind blue-500 (Steering / Active)
COLOR_ACCENT_YELLOW = (245, 158, 11)   # Tailwind amber-500 (Neutral / Caution)

COLOR_KEY_INACTIVE = (71, 85, 105)     # Tailwind slate-600

# ==============================================================================
# Pygame GUI Utility Functions
# ==============================================================================
def draw_key(screen, rect, text, active, active_color, font):
    """Draw a stylized keyboard key indicator."""
    color = active_color if active else COLOR_KEY_INACTIVE
    # Draw key background
    pygame.draw.rect(screen, color, rect, border_radius=6)
    # Draw key border for contrast
    pygame.draw.rect(screen, COLOR_BORDER, rect, width=1, border_radius=6)
    # Draw key text
    text_surf = font.render(text, True, COLOR_TEXT_PRIMARY)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)

def get_font(name_list, size):
    """Attempt to load a preferred font, falling back to system defaults."""
    for name in name_list:
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue
    return pygame.font.Font(None, size)

# GPS telemetry variables
gps = None
gps_speed = 0.0
gps_coords = [0.0, 0.0, 0.0]

def compute_gps_speed():
    global gps_speed, gps_coords
    if gps is None:
        return
    coords = gps.getValues()
    speed_ms = gps.getSpeed()
    gps_speed = speed_ms * 3.6  # convert from m/s to km/h
    print(gps_speed)
    gps_coords = list(coords)

# ==============================================================================
# Main Controller & Application Loop
# ==============================================================================
def main():
    # Initialize Webots vehicle Driver
    print("Connecting to Webots simulation...")
    try:
        driver = Driver()
    except Exception as e:
        print(f"Failed to initialize Webots Driver connection: {e}", file=sys.stderr)
        print("Please ensure that Webots is running and a vehicle model (with controller '<extern>') is loaded.", file=sys.stderr)
        sys.exit(1)
        
    timestep = int(driver.getBasicTimeStep())
    print(f"Connected to Webots. Timestep: {timestep}ms")

    # initialize gps
    global gps
    gps = driver.getDevice("gps")
    if gps is not None:
        gps.enable(timestep)

    # Initial states
    throttle = 0.0
    brake = 0.0
    steering_angle = 0.0
    gear = 1  # 1: Forward, 0: Neutral, -1: Reverse
    desired_speed = 40
    speed_history = []

    # Limits and Increments (tuned for smooth driving)
    max_steering = 0.5            # radians (approx 28.6 degrees)
    steer_rate = 1.2              # rad/s steering speed
    throttle_rate = 1.5           # throttle unit change per second
    brake_rate = 2.5              # brake unit change per second
    
    # Configure initial vehicle settings
    driver.setGear(gear)
    driver.setCruisingSpeed(0.0)  # Cruising speed set to 0 to enable manual throttle control
    driver.setThrottle(throttle)
    driver.setBrakeIntensity(brake)
    driver.setSteeringAngle(steering_angle)

    # Initialize Pygame
    pygame.init()
    window_width, window_height = 680, 520
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Webots Manual Vehicle Control (Direct API)")
    clock = pygame.time.Clock()

    # Load Fonts
    pref_fonts = ["Segoe UI", "Helvetica", "Arial", "sans-serif"]
    font_title = get_font(pref_fonts, 28)
    font_subtitle = get_font(pref_fonts, 16)
    font_large_val = get_font(pref_fonts, 48)
    font_label = get_font(pref_fonts, 14)
    font_value = get_font(pref_fonts, 22)
    font_key = get_font(pref_fonts, 18)

    print("Pygame GUI initialized. Ready for keyboard input.")
    print("Controls:\n"
          "  W / UP Arrow    : Apply Throttle\n"
          "  S / DOWN Arrow  : Apply Brake\n"
          "  A / LEFT Arrow  : Steer Left\n"
          "  D / RIGHT Arrow : Steer Right\n"
          "  G               : Cycle Gears (Forward [1], Neutral [0], Reverse [-1])\n"
          "  SPACEBAR        : Emergency Stop (Brake + Center Steering)\n")

    running = True
    last_time = pygame.time.get_ticks()

    while running:
        # 1. Advance Webots Simulation Step
        # The step function advances simulation time and updates vehicle sensors
        webots_status = driver.step()
        if webots_status == -1:
            print("Webots simulation stopped. Exiting.")
            break

        # Calculate time delta for smooth dynamics integration
        current_time = pygame.time.get_ticks()
        dt = (current_time - last_time) / 1000.0
        last_time = current_time
        if dt <= 0:
            dt = 0.02  # fallback

        # 2. Pygame Event Handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Cycle gear on key press (prevents rapid cycling if key is held)
                if event.key == pygame.K_g:
                    if gear == 1:
                        gear = 0
                    elif gear == 0:
                        gear = -1
                    else:
                        gear = 1
                    driver.setGear(gear)
                    print(f"Gear shifted to: {gear}")
                # Escape key to exit
                elif event.key == pygame.K_ESCAPE:
                    running = False
                # UP or W: increment throttle by 0.05, or decrement brake if brake is active
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    if brake > 0.0:
                        brake = max(0.0, brake - 0.05)
                    else:
                        throttle = min(1.0, throttle + 0.05)
                    print(f"Key UP/W pressed. Throttle: {throttle:.2f}, Brake: {brake:.2f}")
                # DOWN or S: decrement throttle by 0.05, or increment brake if throttle is 0
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    if throttle > 0.0:
                        throttle = max(0.0, throttle - 0.05)
                    else:
                        brake = min(1.0, brake + 0.05)
                    print(f"Key DOWN/S pressed. Throttle: {throttle:.2f}, Brake: {brake:.2f}")
                # C: reset throttle and brake to 0
                elif event.key == pygame.K_c:
                    throttle = 0.0
                    brake = 0.0
                    print(f"Key C pressed. Throttle reset to: {throttle:.2f}, Brake reset to: {brake:.2f}")

        # Get state of all keyboard keys
        keys = pygame.key.get_pressed()

        # 3. Apply Keyboard Input to Steering Dynamics
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering_angle -= steer_rate * dt
            if steering_angle < -max_steering:
                steering_angle = -max_steering
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering_angle += steer_rate * dt
            if steering_angle > max_steering:
                steering_angle = max_steering
        else:
            # Auto-centering steering
            if steering_angle > 0.0:
                steering_angle = max(0.0, steering_angle - steer_rate * dt)
            elif steering_angle < 0.0:
                steering_angle = min(0.0, steering_angle + steer_rate * dt)

        # 4. Apply Keyboard Input to Throttle & Brake Dynamics
        if keys[pygame.K_SPACE]:
            # Emergency Stop
            throttle = 0.0
            brake = 1.0
            steering_angle = 0.0

        # 5. Send Control Inputs to Webots Vehicle
        driver.setThrottle(throttle)
        driver.setBrakeIntensity(brake)
        driver.setSteeringAngle(steering_angle)

        # 6. Retrieve Telemetry Feedback from Webots
        if gps is not None:
            compute_gps_speed()
        current_speed_kmh = driver.getCurrentSpeed()  # Speed returned in km/h
        speed_val = current_speed_kmh if not math.isnan(current_speed_kmh) else 0.0
        speed_history.append(speed_val)
        current_speed_mps = current_speed_kmh / 3.6    # Convert to m/s
        current_rpm = driver.getRpm()
        sim_time = driver.getTime()

        # 7. Render GUI
        screen.fill(COLOR_BG)

        # Header Block
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(20, 20, 640, 60), border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, pygame.Rect(20, 20, 640, 60), width=1, border_radius=8)
        
        title_surf = font_title.render("WEBOTS VEHICLE TEST PANEL", True, COLOR_TEXT_PRIMARY)
        screen.blit(title_surf, (35, 30))
        
        status_text = f"SIM TIME: {sim_time:.2f} s  |  DIRECT CONNECTION ACTIVE"
        status_surf = font_subtitle.render(status_text, True, COLOR_ACCENT_GREEN)
        screen.blit(status_surf, (370, 40))

        # Main Panel Layout: Split Left (Speed & Gear) and Right (Throttle, Brake & Steering)
        panel_left_rect = pygame.Rect(20, 95, 310, 240)
        panel_right_rect = pygame.Rect(350, 95, 310, 240)

        pygame.draw.rect(screen, COLOR_PANEL, panel_left_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, panel_left_rect, width=1, border_radius=8)
        
        pygame.draw.rect(screen, COLOR_PANEL, panel_right_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, panel_right_rect, width=1, border_radius=8)

        # ---- LEFT PANEL: SPEEDOMETER & GEARS ----
        # Speed display
        lbl_speed = font_label.render("VEHICLE SPEED", True, COLOR_TEXT_MUTED)
        panel_left_rect.x += 20
        screen.blit(lbl_speed, (panel_left_rect.x, 110))
        
        speed_str = f"{current_speed_kmh:.1f}"
        val_speed = font_large_val.render(speed_str, True, COLOR_TEXT_PRIMARY)
        screen.blit(val_speed, (panel_left_rect.x, 125))
        
        unit_speed = font_subtitle.render("km/h", True, COLOR_TEXT_MUTED)
        screen.blit(unit_speed, (panel_left_rect.x + val_speed.get_width() + 5, 150))
        
        speed_mps_str = f"({current_speed_mps:.2f} m/s)"
        val_speed_mps = font_subtitle.render(speed_mps_str, True, COLOR_TEXT_MUTED)
        screen.blit(val_speed_mps, (panel_left_rect.x, 185))

        # GPS Speed display
        lbl_gps_speed = font_label.render("GPS SPEED", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_gps_speed, (panel_left_rect.x + 140, 110))
        
        # Check if gps_speed is nan or not initialized
        if math.isnan(gps_speed):
            gps_speed_str = "N/A"
        else:
            gps_speed_str = f"{gps_speed:.1f}"
        val_gps_speed = font_large_val.render(gps_speed_str, True, COLOR_TEXT_PRIMARY)
        screen.blit(val_gps_speed, (panel_left_rect.x + 140, 125))
        
        unit_gps_speed = font_subtitle.render("km/h", True, COLOR_TEXT_MUTED)
        screen.blit(unit_gps_speed, (panel_left_rect.x + 140 + val_gps_speed.get_width() + 5, 150))
        
        if math.isnan(gps_speed):
            gps_speed_mps_str = "(N/A m/s)"
        else:
            gps_speed_mps_str = f"({(gps_speed / 3.6):.2f} m/s)"
        val_gps_speed_mps = font_subtitle.render(gps_speed_mps_str, True, COLOR_TEXT_MUTED)
        screen.blit(val_gps_speed_mps, (panel_left_rect.x + 140, 185))

        # Speed limit indicator bar
        pygame.draw.rect(screen, COLOR_BORDER, pygame.Rect(panel_left_rect.x, 210, 270, 8), border_radius=2)
        # Normalize relative to top speed for visual scaling (e.g. 50 km/h)
        speed_ratio = min(1.0, abs(current_speed_kmh) / 50.0)
        speed_bar_w = int(270 * speed_ratio)
        speed_bar_color = COLOR_ACCENT_GREEN if current_speed_kmh >= 0 else COLOR_ACCENT_RED
        if speed_bar_w > 0:
            pygame.draw.rect(screen, speed_bar_color, pygame.Rect(panel_left_rect.x, 210, speed_bar_w, 8), border_radius=2)

        # Gear Selector Visualizer
        lbl_gear = font_label.render("TRANSMISSION GEAR", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_gear, (panel_left_rect.x, 235))
        
        # Draw three gear slots: Reverse, Neutral, Forward
        gear_labels = ["R", "N", "D"]
        gear_values = [-1, 0, 1]
        gear_colors = [COLOR_ACCENT_RED, COLOR_ACCENT_YELLOW, COLOR_ACCENT_GREEN]
        
        for i, (g_lbl, g_val, g_col) in enumerate(zip(gear_labels, gear_values, gear_colors)):
            slot_rect = pygame.Rect(panel_left_rect.x + (i * 55), 255, 45, 35)
            is_active = (gear == g_val)
            bg_col = g_col if is_active else COLOR_BORDER
            pygame.draw.rect(screen, bg_col, slot_rect, border_radius=4)
            
            txt_color = COLOR_BG if is_active else COLOR_TEXT_PRIMARY
            gear_txt_surf = font_value.render(g_lbl, True, txt_color)
            gear_txt_rect = gear_txt_surf.get_rect(center=slot_rect.center)
            screen.blit(gear_txt_surf, gear_txt_rect)

        # Engine RPM
        rpm_text = f"ENGINE: {current_rpm:.0f} RPM"
        rpm_surf = font_subtitle.render(rpm_text, True, COLOR_TEXT_PRIMARY)
        screen.blit(rpm_surf, (panel_left_rect.x, 305))

        # GPS Coordinates
        if gps_coords and not any(math.isnan(c) for c in gps_coords):
            gps_x, gps_y, gps_z = gps_coords[0], gps_coords[1], gps_coords[2]
            coords_text = f"GPS: {gps_x:.1f}, {gps_y:.1f}"
        else:
            coords_text = "GPS: N/A"
        coords_surf = font_subtitle.render(coords_text, True, COLOR_TEXT_MUTED)
        screen.blit(coords_surf, (panel_left_rect.x + 140, 305))

        # Reset panel left offset
        panel_left_rect.x -= 20

        # ---- RIGHT PANEL: CONTROL TELEMETRY ----
        # Throttle Slider
        lbl_throttle = font_label.render(f"THROTTLE: {throttle*100.0:.0f}%", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_throttle, (370, 110))
        pygame.draw.rect(screen, COLOR_BORDER, pygame.Rect(370, 130, 270, 14), border_radius=3)
        throttle_w = int(270 * throttle)
        if throttle_w > 0:
            pygame.draw.rect(screen, COLOR_ACCENT_GREEN, pygame.Rect(370, 130, throttle_w, 14), border_radius=3)

        # Brake Slider
        lbl_brake = font_label.render(f"BRAKE INTENSITY: {brake*100.0:.0f}%", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_brake, (370, 160))
        pygame.draw.rect(screen, COLOR_BORDER, pygame.Rect(370, 180, 270, 14), border_radius=3)
        brake_w = int(270 * brake)
        if brake_w > 0:
            pygame.draw.rect(screen, COLOR_ACCENT_RED, pygame.Rect(370, 180, brake_w, 14), border_radius=3)

        # Steering Angle
        steer_deg = steering_angle * 180.0 / math.pi
        lbl_steer = font_label.render(f"STEERING ANGLE: {steer_deg:.1f}°", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_steer, (370, 210))
        
        # Center-aligned steering slider
        pygame.draw.rect(screen, COLOR_BORDER, pygame.Rect(370, 230, 270, 14), border_radius=3)
        pygame.draw.line(screen, COLOR_TEXT_MUTED, (505, 227), (505, 247), 2)
        
        steer_ratio = steering_angle / max_steering
        steer_bar_w = int(135 * steer_ratio)
        if steer_bar_w > 0:
            pygame.draw.rect(screen, COLOR_ACCENT_BLUE, pygame.Rect(505, 230, steer_bar_w, 14), border_radius=3)
        elif steer_bar_w < 0:
            pygame.draw.rect(screen, COLOR_ACCENT_BLUE, pygame.Rect(505 + steer_bar_w, 230, -steer_bar_w, 14), border_radius=3)

        # Active Values Text Feedback
        debug_info = f"Throttle (Raw): {throttle:.3f} | Brake (Raw): {brake:.3f}"
        debug_surf = font_subtitle.render(debug_info, True, COLOR_TEXT_MUTED)
        screen.blit(debug_surf, (370, 305))

        # ---- KEYBOARD CONTROLS VISUALIZER ----
        panel_bottom_rect = pygame.Rect(20, 350, 640, 150)
        pygame.draw.rect(screen, COLOR_PANEL, panel_bottom_rect, border_radius=8)
        pygame.draw.rect(screen, COLOR_BORDER, panel_bottom_rect, width=1, border_radius=8)

        lbl_keyboard = font_label.render("CONTROL BUTTON STATE & LEGEND", True, COLOR_TEXT_MUTED)
        screen.blit(lbl_keyboard, (35, 360))

        # Get key active states
        w_active = keys[pygame.K_UP] or keys[pygame.K_w]
        s_active = keys[pygame.K_DOWN] or keys[pygame.K_s]
        a_active = keys[pygame.K_LEFT] or keys[pygame.K_a]
        d_active = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        space_active = keys[pygame.K_SPACE]
        g_active = keys[pygame.K_g]

        # Draw visual keys
        draw_key(screen, pygame.Rect(95, 385, 50, 45), "W", w_active, COLOR_ACCENT_GREEN, font_key)
        draw_key(screen, pygame.Rect(40, 435, 50, 45), "A", a_active, COLOR_ACCENT_BLUE, font_key)
        draw_key(screen, pygame.Rect(95, 435, 50, 45), "S", s_active, COLOR_ACCENT_RED, font_key)
        draw_key(screen, pygame.Rect(150, 435, 50, 45), "D", d_active, COLOR_ACCENT_BLUE, font_key)

        draw_key(screen, pygame.Rect(220, 435, 230, 45), "SPACEBAR  (BRAKE / RESET)", space_active, COLOR_ACCENT_RED, font_key)
        draw_key(screen, pygame.Rect(460, 435, 180, 45), "G  (CYCLE GEAR)", g_active, COLOR_ACCENT_YELLOW, font_key)

        # Legend instructions text
        inst1 = font_subtitle.render("W / S : Accelerate / Brake", True, COLOR_TEXT_PRIMARY)
        inst2 = font_subtitle.render("A / D : Steer Left / Right", True, COLOR_TEXT_PRIMARY)
        screen.blit(inst1, (225, 385))
        screen.blit(inst2, (465, 385))

        # Flip screen buffer and cap loop rate at 60 FPS
        pygame.display.flip()
        clock.tick(60)

    # 8. Clean Exit & Reset Webots Values
    print("Exiting manual control. Resetting vehicle actuators.")
    try:
        driver.setThrottle(0.0)
        driver.setBrakeIntensity(1.0)
        driver.setSteeringAngle(0.0)
        # Advance simulation one step to apply reset values
        driver.step()
    except Exception:
        pass

    pygame.quit()

    if speed_history:
        time_step_sec = timestep / 1000.0
        time_axis = [i * time_step_sec for i in range(len(speed_history))]
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_axis, speed_history, label='Current Velocity', color='blue', linewidth=2)
        plt.axhline(y=desired_speed, color='red', linestyle='--', label=f'Desired Speed ({desired_speed} km/h)')
        plt.title('Vehicle Velocity over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Velocity (km/h)')
        plt.legend()
        plt.grid(True)
        plt.show()

    sys.exit(0)

if __name__ == '__main__':
    main()
