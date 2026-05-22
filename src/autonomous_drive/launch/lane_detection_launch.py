from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='model.pt',
        description='Path to TorchScript model'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/vehicle/camera/image_color',
        description='Camera image topic'
    )

    lane_detection_node = Node(
        package='autonomous_drive',
        executable='lane_detection',
        name='lane_detection',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
            }
        ]
    )

    return LaunchDescription([
        model_path_arg,
        image_topic_arg,
        lane_detection_node,
    ])