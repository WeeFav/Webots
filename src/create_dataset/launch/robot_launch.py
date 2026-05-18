import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController
import cv2

def generate_launch_description():
    package_dir = get_package_share_directory('create_dataset')
    robot_description_path_0 = os.path.join(package_dir, 'resource', 'SUMO_VEHICLE0.urdf')
    robot_description_path_1 = os.path.join(package_dir, 'resource', 'SUMO_VEHICLE1.urdf')

    robot_driver_0 = WebotsController(
        robot_name='SUMO_VEHICLE0',
        parameters=[
            {'robot_description': robot_description_path_0},
            {'use_sim_time': True},
            {'saving': True},
            {'data_root': "/home/marvin/Webots/Webots-Lane-Detection/datasets/normal_city"},
            {'save_freq': 90},
            {'save_num': 250},
        ]
    )
    
    # robot_driver_1 = WebotsController(
    #     robot_name='SUMO_VEHICLE1',
    #     parameters=[
    #         {'robot_description': robot_description_path_1},
    #         {'use_sim_time': True},
    #     ]
    # )

    return LaunchDescription([
        robot_driver_0,
        # robot_driver_1,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=robot_driver_0,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
        # launch.actions.RegisterEventHandler(
        #     event_handler=launch.event_handlers.OnProcessExit(
        #         target_action=robot_driver_1,
        #         on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
        #     )
        # ),
    ])