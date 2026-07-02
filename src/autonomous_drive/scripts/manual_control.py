#!/usr/bin/env python3
import sys
import math
import pygame
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive

# UI Constants
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 400
COLOR_BG = (13, 17, 23)
COLOR_PANEL = (22, 27, 34)
COLOR_TEXT_PRIMARY = (240, 246, 252)
COLOR_TEXT_MUTED = (139, 148, 158)
COLOR_ACCENT_ACTIVE = (88, 166, 255)
COLOR_ACCENT_GREEN = (57, 255, 20)
COLOR_ACCENT_RED = (255, 123, 114)
COLOR_KEY_INACTIVE = (48, 54, 61)

class ManualControlNode(Node):
    def __init__(self):
        super().__init__('manual_control')
        
        # Declare parameters for vehicle dynamics
        self.declare_parameter('top_speed', 40.0)             # km/h
        self.declare_parameter('time_to_top_speed', 1.0)      # seconds
        self.declare_parameter('max_steering', 0.5)           # radians (approx 28.6 degrees)
        self.declare_parameter('time_to_max_steering', 1.0)   # seconds
        
        # Get parameter values
        self.top_speed = self.get_parameter('top_speed').get_parameter_value().double_value
        self.time_to_top_speed = self.get_parameter('time_to_top_speed').get_parameter_value().double_value
        self.max_steering = self.get_parameter('max_steering').get_parameter_value().double_value
        self.time_to_max_steering = self.get_parameter('time_to_max_steering').get_parameter_value().double_value
        
        # Create Publisher
        self.publisher = self.create_publisher(AckermannDrive, '/cmd_ackermann', 10)
        
        # Internal State Variables
        self.speed_kmh = 0.0
        self.steering_rad = 0.0
        
        # Calculate dynamic rates (per second)
        self.accel_rate = self.top_speed / self.time_to_top_speed if self.time_to_top_speed > 0 else self.top_speed
        self.turn_rate = self.max_steering / self.time_to_max_steering if self.time_to_max_steering > 0 else self.max_steering
        
        self.get_logger().info(
            f"Pygame Control Node initialized.\n"
            f"  Top Speed: {self.top_speed} km/h (Accel Time: {self.time_to_top_speed}s)\n"
            f"  Max Steering: {self.max_steering} rad (Steer Time: {self.time_to_max_steering}s)"
        )

    def update_dynamics(self, keys, dt):
        """Update speed and steering based on keyboard state and delta time."""
        # 1. Speed Dynamics
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            # Accelerate forward
            self.speed_kmh += self.accel_rate * dt
            if self.speed_kmh > self.top_speed:
                self.speed_kmh = self.top_speed
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            # Accelerate backwards / Decelerate
            self.speed_kmh -= self.accel_rate * dt
            if self.speed_kmh < -self.top_speed:
                self.speed_kmh = -self.top_speed
        else:
            # Let go: decay to 0
            if self.speed_kmh > 0:
                self.speed_kmh = max(0.0, self.speed_kmh - self.accel_rate * dt)
            elif self.speed_kmh < 0:
                self.speed_kmh = min(0.0, self.speed_kmh + self.accel_rate * dt)
                
        # 2. Steering Dynamics
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            # Turn Left
            self.steering_rad -= self.turn_rate * dt
            if self.steering_rad < -self.max_steering:
                self.steering_rad = -self.max_steering
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            # Turn Right
            self.steering_rad += self.turn_rate * dt
            if self.steering_rad > self.max_steering:
                self.steering_rad = self.max_steering
        else:
            # Let go: auto-center back to 0
            if self.steering_rad > 0:
                self.steering_rad = max(0.0, self.steering_rad - self.turn_rate * dt)
            elif self.steering_rad < 0:
                self.steering_rad = min(0.0, self.steering_rad + self.turn_rate * dt)
                
        # 3. Emergency Brake / Reset
        if keys[pygame.K_SPACE]:
            self.speed_kmh = 0.0
            self.steering_rad = 0.0

    def publish_command(self):
        """Publish the current active control command."""
        msg = AckermannDrive()
        msg.speed = self.speed_kmh / 3.6  # convert km/h to m/s
        msg.steering_angle = self.steering_rad
        self.publisher.publish(msg)


def draw_key(screen, rect, text, active, active_color, font):
    """Utility to draw a stylized key indicator."""
    color = active_color if active else COLOR_KEY_INACTIVE
    pygame.draw.rect(screen, color, rect, border_radius=5)
    # Text
    text_surf = font.render(text, True, COLOR_TEXT_PRIMARY)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)


def main(args=None):
    rclpy.init(args=args)
    node = ManualControlNode()
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("ROS2 Vehicle Manual Control")
    clock = pygame.time.Clock()
    
    # Load fonts
    font_large = pygame.font.Font(None, 36)
    font_value = pygame.font.Font(None, 44)
    font_small = pygame.font.Font(None, 20)
    font_key = pygame.font.Font(None, 24)
    
    running = True
    while running:
        # Calculate dt in seconds
        dt = clock.tick(50) / 1000.0  # limit loop to 50 Hz
        
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # Get active keys
        keys = pygame.key.get_pressed()
        
        # Update vehicle speed and steering kinematics
        node.update_dynamics(keys, dt)
        
        # Publish ROS2 Ackermann command
        node.publish_command()
        
        # Spin ROS2 callbacks
        rclpy.spin_once(node, timeout_sec=0.0)
        
        # ---- Render Pygame UI ----
        screen.fill(COLOR_BG)
        
        # 1. Draw Title
        title_surf = font_large.render("VEHICLE CONTROL INTERFACE", True, COLOR_TEXT_PRIMARY)
        screen.blit(title_surf, (20, 20))
        
        # 2. Draw Gauges Panel
        pygame.draw.rect(screen, COLOR_PANEL, pygame.Rect(20, 70, 460, 160), border_radius=8)
        
        # Speed Value Display
        speed_text = f"{node.speed_kmh:.1f} km/h"
        speed_mps_text = f"({(node.speed_kmh/3.6):.2f} m/s)"
        lbl_speed = font_small.render("CURRENT SPEED", True, COLOR_TEXT_MUTED)
        val_speed = font_value.render(speed_text, True, COLOR_TEXT_PRIMARY)
        val_speed_mps = font_small.render(speed_mps_text, True, COLOR_TEXT_MUTED)
        
        screen.blit(lbl_speed, (40, 85))
        screen.blit(val_speed, (40, 105))
        screen.blit(val_speed_mps, (200, 118))
        
        # Speed Progress Bar Gauge
        pygame.draw.rect(screen, COLOR_KEY_INACTIVE, pygame.Rect(40, 140, 420, 14), border_radius=3)
        # Fill proportion
        speed_ratio = abs(node.speed_kmh) / max(1.0, node.top_speed)
        fill_width = int(420 * speed_ratio)
        fill_color = COLOR_ACCENT_GREEN if node.speed_kmh >= 0 else COLOR_ACCENT_RED
        if fill_width > 0:
            pygame.draw.rect(screen, fill_color, pygame.Rect(40, 140, fill_width, 14), border_radius=3)
            
        # Steering Value Display
        steering_deg = node.steering_rad * 180.0 / math.pi
        steer_text = f"{abs(steering_deg):.1f}° {'Right' if steering_deg > 0 else 'Left' if steering_deg < 0 else 'Center'}"
        lbl_steer = font_small.render("STEERING ANGLE", True, COLOR_TEXT_MUTED)
        val_steer = font_value.render(steer_text, True, COLOR_TEXT_PRIMARY)
        
        screen.blit(lbl_steer, (40, 170))
        screen.blit(val_steer, (40, 190))
        
        # Steering Center Bar Gauge
        pygame.draw.rect(screen, COLOR_KEY_INACTIVE, pygame.Rect(260, 192, 180, 14), border_radius=3)
        # Center line
        pygame.draw.line(screen, COLOR_TEXT_MUTED, (350, 190), (350, 208), 2)
        # Steer fill relative to center (350 is center width of 180 bar starting at 260)
        steer_ratio = node.steering_rad / max(0.01, node.max_steering)
        steer_fill_w = int(90 * steer_ratio)
        if steer_fill_w > 0:
            pygame.draw.rect(screen, COLOR_ACCENT_ACTIVE, pygame.Rect(350, 192, steer_fill_w, 14), border_radius=3)
        elif steer_fill_w < 0:
            pygame.draw.rect(screen, COLOR_ACCENT_ACTIVE, pygame.Rect(350 + steer_fill_w, 192, -steer_fill_w, 14), border_radius=3)
            
        # 3. Draw Keyboard Visualizer
        # Layout positions for W, A, S, D / Arrows
        up_active = keys[pygame.K_UP] or keys[pygame.K_w]
        down_active = keys[pygame.K_DOWN] or keys[pygame.K_s]
        left_active = keys[pygame.K_LEFT] or keys[pygame.K_a]
        right_active = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        space_active = keys[pygame.K_SPACE]
        
        # Draw UP key
        draw_key(screen, pygame.Rect(95, 250, 50, 40), "W / \u2191", up_active, COLOR_ACCENT_GREEN, font_key)
        # Draw LEFT key
        draw_key(screen, pygame.Rect(40, 295, 50, 40), "A / \u2190", left_active, COLOR_ACCENT_ACTIVE, font_key)
        # Draw DOWN key
        draw_key(screen, pygame.Rect(95, 295, 50, 40), "S / \u2193", down_active, COLOR_ACCENT_RED, font_key)
        # Draw RIGHT key
        draw_key(screen, pygame.Rect(150, 295, 50, 40), "D / \u2192", right_active, COLOR_ACCENT_ACTIVE, font_key)
        # Draw SPACE key (Brake)
        draw_key(screen, pygame.Rect(220, 295, 240, 40), "SPACEBAR  (BRAKE)", space_active, COLOR_ACCENT_RED, font_key)
        
        # 4. Info Footer
        info_text = f"Top Speed: {node.top_speed} km/h ({node.time_to_top_speed}s)  |  Max Turn: {node.max_steering:.2f} rad ({node.time_to_max_steering}s)"
        info_surf = font_small.render(info_text, True, COLOR_TEXT_MUTED)
        screen.blit(info_surf, (20, 365))
        
        pygame.display.flip()
        
    # Send final stop command
    node.get_logger().info("Exiting. Sending stop command.")
    stop_msg = AckermannDrive()
    stop_msg.speed = 0.0
    stop_msg.steering_angle = 0.0
    node.publisher.publish(stop_msg)
    
    # Shutdown
    pygame.quit()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
