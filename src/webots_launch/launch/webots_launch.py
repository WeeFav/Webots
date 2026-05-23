import os
import launch

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_launcher import WebotsLauncher


def launch_setup(context, *args, **kwargs):
    world = LaunchConfiguration('world').perform(context)

    webots = WebotsLauncher(
        world=world,
        mode='pause',
        ros2_supervisor=True
    )

    return [
        webots,
        webots._supervisor,

        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=webots,
                on_exit=[
                    launch.actions.EmitEvent(
                        event=launch.events.Shutdown()
                    )
                ],
            )
        ),
    ]


def generate_launch_description():
    package_dir = get_package_share_directory('webots_launch')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='city.wbt',
            description='Path to world file'
        ),

        OpaqueFunction(function=launch_setup)
    ])