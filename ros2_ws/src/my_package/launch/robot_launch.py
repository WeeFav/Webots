import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController
import cv2

def generate_launch_description():
    package_dir = get_package_share_directory('my_package')
    robot_description_path = os.path.join(package_dir, 'resource', 'my_robot.urdf')

    my_robot_driver = WebotsController(
        robot_name='vehicle',
        parameters=[
            {'robot_description': robot_description_path},
        ]
    )

    lane_follower = Node(
        package='my_package',
        executable='lane_follower',
    )

    return LaunchDescription([
        my_robot_driver,
        lane_follower,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=my_robot_driver,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=lane_follower,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        )
    ])