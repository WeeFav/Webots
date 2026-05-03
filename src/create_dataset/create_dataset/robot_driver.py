import rclpy
from ackermann_msgs.msg import AckermannDrive
import rclpy.logging
import numpy as np
import cv2
import sys
import os
from create_dataset.lane_segmentation import extract_lanes
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header

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
        self.lidar_node = self.__robot.getFromDef("LIDAR")
        
        self.lidar = self.__robot.getDevice("lidar")
        self.lidar.enable(32)
        self.lidar.enablePointCloud()
        
        self.__node.create_subscription(AckermannDrive, 'cmd_ackermann', self.__cmd_ackermann_callback, 1)
        self.lane_seg_pub = self.__node.create_publisher(Image, 'segmentation', 10)
        self.obj_det_pub = self.__node.create_publisher(MarkerArray, "bbox_markers", 10)
        self.lidar_pub = self.__node.create_publisher(PointCloud2, "/points", 10)
        self.bridge = CvBridge()
        rclpy.get_default_context().on_shutdown(self.cleanup)
    
        world_path = "/home/marvin/Webots/src/create_dataset/worlds/city.wbt"
        self.all_lanes_center = extract_lanes(world_path)

        self.vehicles = {}
        for i in range(100):
            defName = "SUMO_VEHICLE%d" % i
            node = self.__robot.getFromDef(defName)
            if node:
                self.vehicles[i] = node
            else:
                break
            
        self.edges = [
            (0,1), (1,2), (2,3), (3,0),  # bottom face
            (4,5), (5,6), (6,7), (7,4),  # top face
            (0,4), (1,5), (2,6), (3,7)   # vertical edges
        ]
        
        self.step_count = 0
        
    def cleanup(self, signum, frame):
        print("Ctrl+C detected, closing windows...")
        cv2.destroyAllWindows()
        sys.exit(0)
        
    def __cmd_ackermann_callback(self, message):
        self.__robot.setCruisingSpeed(message.speed)
        self.__robot.setSteeringAngle(message.steering_angle)
        
    def lane_detection(self):
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
        
        # cv2.imshow("segmentation", img)
        # cv2.waitKey(1)
        
        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        self.lane_seg_pub.publish(msg)
        
    def object_detection(self):
        t_lidar_to_world = np.array(self.lidar_node.getPosition())
        R_liar_to_world = np.array(self.lidar_node.getOrientation()).reshape(3, 3)
        
        all_corners = []
        for i, vehicle in self.vehicles.items():
            t_vehicle_to_world = np.array(vehicle.getPosition()) # world coordinate
            R_vehicle_to_world = np.array(vehicle.getOrientation()).reshape(3, 3)
            
            boundingObject = vehicle.getBaseNodeField("boundingObject").getSFNode()
            boxes = self.extract_boxes(boundingObject)
            corners_vehicle = self.get_bounding_box(boxes)
            
            corners_world = (R_vehicle_to_world @ corners_vehicle.T).T + t_vehicle_to_world
            corners_lidar = (R_liar_to_world.T @ (corners_world - t_lidar_to_world).T).T
            
            all_corners.append(corners_lidar)
            
        marker_array = self.corners_to_marker_array(all_corners)
        self.obj_det_pub.publish(marker_array)

    def extract_boxes(self, node, parent_R=np.eye(3), parent_t=np.zeros(3)):
        boxes = [] # hold all boxes found under this node
        
        node_type = node.getTypeName()

        if node_type == "Group":
            for i in range(node.getField("children").getCount()):
                child = node.getField("children").getMFNode(i)
                boxes += self.extract_boxes(child, parent_R, parent_t)
        elif node_type == "Pose":
            t = np.array(node.getField("translation").getSFVec3f())
            rot = node.getField("rotation").getSFRotation()

            # axis-angle → rotation matrix
            axis = np.array(rot[:3])
            angle = rot[3]
            rot = Rotation.from_rotvec(np.array(axis) * angle)
            R = rot.as_matrix()
            
            new_R = parent_R @ R
            new_t = parent_R @ t + parent_t

            for i in range(node.getField("children").getCount()):
                child = node.getField("children").getMFNode(i)
                boxes += self.extract_boxes(child, new_R, new_t)
        elif node_type == "Box":
            size = np.array(node.getField("size").getSFVec3f())
            boxes.append((parent_R, parent_t, size))

        return boxes

    def get_box_corners(self, box):
        R_local_to_vehicle = box[0]
        t_local_to_vehicle = box[1]
        size = box[2]
        x, y, z = size / 2.0    
        corners_local = np.array([
            [ x, -y, -z],   # front-left-bottom
            [ x,  y, -z],   # front-right-bottom
            [-x,  y, -z],   # rear-right-bottom
            [-x, -y, -z],   # rear-left-bottom
            [ x, -y,  z],   # front-left-top
            [ x,  y,  z],   # front-right-top
            [-x,  y,  z],   # rear-right-top
            [-x, -y,  z],   # rear-left-top
        ])
        
        corners_vehicle = (R_local_to_vehicle @ corners_local.T).T + t_local_to_vehicle   
        return corners_vehicle
        
    def get_bounding_box(self, boxes):
        all_corners = []

        for box in boxes:
            corners = self.get_box_corners(box)
            all_corners.append(corners)

        all_corners = np.vstack(all_corners) # shape: (N*8, 3)    

        min_corner = np.min(all_corners, axis=0)
        max_corner = np.max(all_corners, axis=0)

        center = (min_corner + max_corner) / 2
        size = max_corner - min_corner

        x, y, z = size / 2.0

        corners = np.array([
            [ x, -y, -z],
            [ x,  y, -z],
            [-x,  y, -z],
            [-x, -y, -z],
            [ x, -y,  z],
            [ x,  y,  z],
            [-x,  y,  z],
            [-x, -y,  z],
        ])
        
        return corners + center

    def corners_to_marker_array(self, all_corners):
        marker_array = MarkerArray()

        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]

        for obj_id, corners in enumerate(all_corners):

            marker = Marker()
            marker.header.frame_id = "lidar"
            marker.header.stamp = self.__node.get_clock().now().to_msg()

            marker.ns = "bounding_boxes"
            marker.id = obj_id
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD

            marker.scale.x = 0.05  # line width

            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker.points = []

            for start, end in edges:
                p1 = Point()
                p1.x, p1.y, p1.z = corners[start]

                p2 = Point()
                p2.x, p2.y, p2.z = corners[end]

                marker.points.append(p1)
                marker.points.append(p2)

            marker_array.markers.append(marker)

        return marker_array

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        self.step_count += 1
        
        if self.step_count > 2:
            points = self.lidar.getPointCloud()
        
            if points:
                points = np.array([(point.x, point.y, point.z) for point in points], dtype=np.float32)

                # reshape if needed (Nx3)
                if points.ndim == 1:
                    points = points.reshape(-1, 3)  
                    
                header = Header()
                header.stamp = self.__node.get_clock().now().to_msg()
                header.frame_id = "lidar"

                msg = pc2.create_cloud_xyz32(header, points.tolist())

                self.lidar_pub.publish(msg)            
                  
        self.lane_detection()
        self.object_detection()           
