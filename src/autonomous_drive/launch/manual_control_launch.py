from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    manual_control_node = Node(
        package='autonomous_drive',
        executable='manual_control',
        name='manual_control',
        output='screen'
    )

    return LaunchDescription([
        manual_control_node,
    ])
