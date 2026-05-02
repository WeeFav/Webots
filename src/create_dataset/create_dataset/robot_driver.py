import rclpy
from ackermann_msgs.msg import AckermannDrive
import rclpy.logging
import numpy as np
import cv2
import sys
import os
from create_dataset.lane_segmentation import extract_lanes

def get_intrinsic_matrix(width, height, fov):
    fx = width / (2 * np.tan(fov / 2))
    fy = fx
    cx = width / 2
    cy = height / 2

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ])
    return K

def get_extrinsic_matrix(position, orientation):
    R = np.array(orientation).reshape(3, 3)
    t = np.array(position).reshape(3, 1)

    # World → camera
    R_wc = R.T
    t_wc = -R_wc @ t

    extrinsic = np.hstack((R_wc, t_wc))
    return extrinsic

def world_to_image(points_world, K, extrinsic, self):
    """
    points_world: (N, 3)
    K: (3, 3)
    extrinsic: (3, 4)

    returns: (N, 2) pixel coordinates
    """

    N = points_world.shape[0]

    # Convert to homogeneous
    points_h = np.hstack((points_world, np.ones((N, 1))))  # (N, 4)
    
    # World to Camera
    # this will transform points from webots world coordinate system to webots camera coordinate system
    # webots world coordinate system and camera coordinate system is x front, y left, z up 
    points_cam = (extrinsic @ points_h.T).T  # (N, 3)

    R_webots_to_opencv = np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ])

    # transform points from webots camera cooridnate system to opencv camera cooridnate system
    # opencv camera cooridnate system is x right, y down, z front
    p_cv = (R_webots_to_opencv @ points_cam.T).T
    
    # filter points in front of camera and not too far
    mask = (p_cv[:, 2] > 1e-6) & (p_cv[:, 2] < 50.0) # avoid division by zero
    p_cv = p_cv[mask]
    
    # Project
    points_img = (K @ p_cv.T).T  # (N, 3)

    # Normalize
    points_img[:, 0] /= points_img[:, 2]
    points_img[:, 1] /= points_img[:, 2]

    return points_img[:, :2]

def interpolate_polyline(points, num_points=100):
    """
    points: (N, 2)
    returns: (num_points, 2)
    """

    if len(points) < 2:
        return points

    # Compute distances between consecutive points
    deltas = np.diff(points, axis=0)
    dists = np.linalg.norm(deltas, axis=1)

    # Cumulative distance
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # Normalize to [0, 1]
    t = cumdist / cumdist[-1]

    # New evenly spaced samples
    t_new = np.linspace(0, 1, num_points)

    # Interpolate x and y separately
    x_new = np.interp(t_new, t, points[:, 0])
    y_new = np.interp(t_new, t, points[:, 1])

    return np.stack([x_new, y_new], axis=1)

class RobotDriver:
    def init(self, webots_node, properties):
        rclpy.init(args=None)
        self.__node = rclpy.create_node('robot_driver_node')
        self.__robot = webots_node.robot
        
        self.camera = self.__robot.getDevice("cam0")
        self.camera.enable(30)
        self.width = self.camera.getWidth()
        self.height = self.camera.getHeight()
        fov = self.camera.getFov()
        
        self.K = get_intrinsic_matrix(self.width, self.height, fov)
        
        self.camera_node = self.__robot.getFromDef("CAMERA")
        
        lidar = self.__robot.getDevice("lidar")
        lidar.enable(100)
        lidar.enablePointCloud()
        
        self.__node.create_subscription(AckermannDrive, 'cmd_ackermann', self.__cmd_ackermann_callback, 1)
        rclpy.get_default_context().on_shutdown(self.cleanup)
    
        world_path = "/home/marvin/Webots/src/create_dataset/worlds/city.wbt"
        self.all_lanes_center = extract_lanes(world_path)
        self.__node.get_logger().info(f"{self.all_lanes_center}")
        
    def cleanup(self, signum, frame):
        print("Ctrl+C detected, closing windows...")
        cv2.destroyAllWindows()
        sys.exit(0)
        
    def __cmd_ackermann_callback(self, message):
        self.__robot.setCruisingSpeed(message.speed)
        self.__robot.setSteeringAngle(message.steering_angle)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
                        
        position = self.camera_node.getPosition()
        orientation = self.camera_node.getOrientation()
        extrinsic = get_extrinsic_matrix(position, orientation)
        
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        for lanes in self.all_lanes_center:
            coords = world_to_image(np.array(lanes), self.K, extrinsic, self)

            # Convert to integer pixel coordinates
            pts = coords.astype(np.int32)

            # Filter valid points inside image
            mask = (
                (pts[:, 0] >= 0) & (pts[:, 0] < self.width) &
                (pts[:, 1] >= 0) & (pts[:, 1] < self.height)
            )
            pts = pts[mask]

            # Need at least 2 points to draw a line
            if len(pts) >= 2:
                pts = pts.reshape((-1, 1, 2))  # required shape for OpenCV
                cv2.polylines(img, [pts], isClosed=False, color=(255, 255, 255), thickness=2)                    
        
        cv2.imshow("segmentation", img)
        cv2.waitKey(1)
