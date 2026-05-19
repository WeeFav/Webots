import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController
import cv2

def generate_launch_description():
    package_dir = get_package_share_directory('create_dataset')
    robot_description_path = os.path.join(package_dir, 'resource', 'robot.urdf')

    robot_driver_0 = WebotsController(
        robot_name='vehicle',
        parameters=[
            {'robot_description': robot_description_path},
            {'use_sim_time': True},
        ]
    )
    
    return LaunchDescription([
        robot_driver_0,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=robot_driver_0,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])