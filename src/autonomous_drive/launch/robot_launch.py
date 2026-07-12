import os
import launch
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    package_dir = get_package_share_directory('autonomous_drive')
    robot_description_path = os.path.join(package_dir, 'resource', 'robot.urdf')

    # Webots Robot Driver
    robot_driver_0 = WebotsController(
        robot_name='WEBOTS_VEHICLE0',
        parameters=[
            {'robot_description': robot_description_path},
            {'use_sim_time': True},
        ]
    )

    # PID Controller
    pid_controller = Node(
        package='autonomous_drive',
        executable='pid_controller',
        name='pid_controller',
        output='screen'
    )

    # Pure Pursuit Controller
    pure_pursuit_controller = Node(
        package='autonomous_drive',
        executable='pure_pursuit_controller',
        name='pure_pursuit_controller',
        output='screen'
    )

    # Lanelet2 RViz Visualizer Launch
    lanelet2_visualizer_pkg = get_package_share_directory('lanelet2_rviz_visualizer')
    visualize_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lanelet2_visualizer_pkg, 'launch', 'visualize.launch.py')
        ),
        launch_arguments={'map_file': '/home/marvin/Webots/map4_sumo_to_lanelet.osm'}.items()
    )

    # Waypoint Publisher
    lanelet2_waypoint_publisher = Node(
        package='autonomous_drive',
        executable='lanelet2_waypoint_publisher',
        name='lanelet2_waypoint_publisher',
        output='screen'
    )

    # RViz
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=[
            '-d',
            '/home/marvin/Webots/src/autonomous_drive/resource/rviz.rviz'
        ],
    )

    return LaunchDescription([
        robot_driver_0,
        pid_controller,
        pure_pursuit_controller,
        visualize_launch,
        lanelet2_waypoint_publisher,
        rviz,
        launch.actions.RegisterEventHandler(
            event_handler=launch.event_handlers.OnProcessExit(
                target_action=robot_driver_0,
                on_exit=[launch.actions.EmitEvent(event=launch.events.Shutdown())],
            )
        ),
    ])