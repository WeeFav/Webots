# Copyright 1996-2023 Cyberbotics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ROS2 Tesla driver."""

import rclpy
from ackermann_msgs.msg import AckermannDrive
import rclpy.logging
from controller import Supervisor
import numpy as np
import cv2
import sys
import signal

class TeslaDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot

        self.camera = self.__robot.getDevice("cam0")
        self.camera.enable(30)
        self.camera.recognitionEnable(30)
        self.camera.enableRecognitionSegmentation()
        
        lidar = self.__robot.getDevice("lidar")
        lidar.enable(100)
        lidar.enablePointCloud()
        
        self.road_node = self.__robot.getFromDef("ROAD_SEGMENT_0")

        # ROS interface
        rclpy.init(args=None)
        self.__node = rclpy.create_node('tesla_node')
        self.__node.create_subscription(AckermannDrive, 'cmd_ackermann', self.__cmd_ackermann_callback, 1)
        
        rclpy.get_default_context().on_shutdown(self.cleanup)
    
    def cleanup(self, signum, frame):
        print("Ctrl+C detected, closing windows...")
        cv2.destroyAllWindows()
        sys.exit(0)
        
    def __cmd_ackermann_callback(self, message):
        self.__robot.setCruisingSpeed(message.speed)
        self.__robot.setSteeringAngle(message.steering_angle)

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
        
        pos = self.road_node.getPosition()
        self.__node.get_logger().info(f"{pos}")
        
        seg = self.camera.getRecognitionSegmentationImage()
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        
        if seg:
            img = np.frombuffer(seg, dtype=np.uint8).reshape((height, width, 4))
            img = img[:, :, :3]  # drop alpha

            cv2.imshow("segmentation", img)
            cv2.waitKey(1)
            