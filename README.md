```
colcon build --symlink-install --packages-select webots_launch autonomous_drive
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
```