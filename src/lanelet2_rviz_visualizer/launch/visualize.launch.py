"""
Launch the Lanelet2 RViz visualizer node.

Usage:
  ros2 launch lanelet2_rviz_visualizer visualize.launch.py \
      map_file:=/path/to/your/map.osm \
      origin_lat:=37.3861 \
      origin_lon:=-122.0839
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    map_file_arg = DeclareLaunchArgument(
        "map_file",
        default_value="map.osm",
        description="Absolute path to the Lanelet2 .osm map file",
    )

    origin_lat_arg = DeclareLaunchArgument(
        "origin_lat",
        default_value="0.0",
        description="Latitude of the UTM projection origin (decimal degrees)",
    )

    origin_lon_arg = DeclareLaunchArgument(
        "origin_lon",
        default_value="0.0",
        description="Longitude of the UTM projection origin (decimal degrees)",
    )

    proj_parameter_arg = DeclareLaunchArgument(
        "proj_parameter",
        default_value="+proj=tmerc +lat_0=25.01291 +lon_0=121.46627 +datum=WGS84 +ellps=WGS84 +units=m +no_defs",
        description="PROJ projection parameter string",
    )

    frame_id_arg = DeclareLaunchArgument(
        "frame_id",
        default_value="map",
        description="TF frame id used for all markers",
    )

    publish_rate_arg = DeclareLaunchArgument(
        "publish_rate_hz",
        default_value="1.0",
        description="How often (Hz) to republish markers (keep alive for RViz)",
    )

    points_x_arg = DeclareLaunchArgument(
        "points_x",
        default_value="[-463.29]",
        description="X coordinates of custom points to visualize as spheres",
    )

    points_y_arg = DeclareLaunchArgument(
        "points_y",
        default_value="[144.96]",
        description="Y coordinates of custom points to visualize as spheres",
    )

    points_z_arg = DeclareLaunchArgument(
        "points_z",
        default_value="[0.05]",
        description="Z coordinates of custom points to visualize as spheres",
    )

    visualizer_node = Node(
        package="lanelet2_rviz_visualizer",
        executable="lanelet2_visualizer",
        name="lanelet2_visualizer",
        output="screen",
        parameters=[{
            "map_file":        LaunchConfiguration("map_file"),
            "origin_lat":      LaunchConfiguration("origin_lat"),
            "origin_lon":      LaunchConfiguration("origin_lon"),
            "projParameter":   LaunchConfiguration("proj_parameter"),
            "frame_id":        LaunchConfiguration("frame_id"),
            "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
            "points_x":        LaunchConfiguration("points_x"),
            "points_y":        LaunchConfiguration("points_y"),
            "points_z":        LaunchConfiguration("points_z"),
        }],
    )

    return LaunchDescription([
        map_file_arg,
        origin_lat_arg,
        origin_lon_arg,
        proj_parameter_arg,
        frame_id_arg,
        publish_rate_arg,
        points_x_arg,
        points_y_arg,
        points_z_arg,
        visualizer_node,
    ])