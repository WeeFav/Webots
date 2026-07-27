```
colcon build --symlink-install --packages-select autonomous_drive
source install/setup.bash
```

```
ros2 launch webots_launch webots_launch.py world:=/home/marvin/Webots/src/webots_launch/worlds/create_dataset/map4.wbt

ros2 launch create_dataset robot_launch.py

ros2 launch lanelet2_rviz_visualizer visualize.launch.py map_file:=/home/marvin/Webots/map4_sumo_to_lanelet.osm
```

```
ros2 launch webots_launch webots_launch.py world:=/home/marvin/Webots/src/webots_launch/worlds/autonomous_drive/map4_robot.wbt

ros2 launch autonomous_drive robot_launch.py

ros2 launch lio_sam run.launch.py

ros2 run autonomous_drive manual_control
```

```
ros2 run autonomous_drive lanelet2_waypoint_publisher
ros2 run autonomous_drive pid_controller
ros2 run autonomous_drive pure_pursuit_controller

2245, 2238, 2231, 4167, 3122, 5723, 3149, 4809, 3202, 3220, 3801, 1639, 4682, 1667, 4541, 1777
```