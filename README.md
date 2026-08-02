Build
```
colcon build --symlink-install --packages-select autonomous_drive
source install/setup.bash
```

Create Dataset
```
ros2 launch webots_launch webots_launch.py world:=/home/marvin/Webots/src/webots_launch/worlds/create_dataset/map4.wbt
ros2 launch create_dataset robot_launch.py
```

Autonomous Drive
```
# map
ros2 launch webots_launch webots_launch.py world:=/home/marvin/Webots/src/webots_launch/worlds/autonomous_drive/map4_robot.wbt

# main control
ros2 launch autonomous_drive robot_launch.py

# SLAM
ros2 launch lio_sam run.launch.py

# manual/autonomous GUI
ros2 run autonomous_drive manual_control
```

Map Creation
```
# save map
ros2 service call /lio_sam/save_map lio_sam/srv/SaveMap "{destination: /Webots/session2_map}"

# merge map (GPS, IMU)
ros2 run autonomous_drive align_pcd_maps \
  --map1 session1_map/GlobalMap.pcd --gps1 25.012562 121.466838 1.557820 \
  --quat1 0.0013875008952695877 0.0009375014607109571 0.854233181330776 0.519819926952773 \
  --map2 session2_map/GlobalMap.pcd --gps2 25.012318 121.466846 1.503477 \
  --quat2 0.001268760589103432 0.001000005029291599 -0.5203799060427668 0.8538268023469364 \
  --out2 map2_aligned.pcd \
  --merged merged_map.pcd

# merge map (lidar ground truth)
ros2 run autonomous_drive align_pcd_maps \
  --map1 session1_map/GlobalMap.pcd \
  --trans1 56.9115 -38.8835 1.56091 \
  --rot1 -2.4194e-06 -0.000792577 1 2.0472 \
  --map2 session2_map/GlobalMap.pcd \
  --trans2 57.3837 -66.2511 1.55936 \
  --rot2 -0.0013009 3.97107e-06 -0.999999 1.0944 \
  --out2 map2_aligned.pcd \
  --merged merged_map.pcd

# visualize SLAM map
ros2 run autonomous_drive pcd_publisher --pcd merged_map.pcd --topic /pcd_map --frame inital_lidar --leaf 0.4

# visualize lanelet2 map
ros2 launch lanelet2_rviz_visualizer visualize.launch.py map_file:=/home/marvin/Webots/map4_sumo_to_lanelet.osm

# tf from SLAM local map to global map
ros2 run autonomous_drive webots_tf_publisher \
  --trans 57.3837 -66.2511 1.55936 \
  --rot -0.0013009 3.97107e-06 -0.999999 1.0944 \
  --frame map --child inital_lidar
```

Others
```
ros2 run autonomous_drive lanelet2_waypoint_publisher
ros2 run autonomous_drive pid_controller
ros2 run autonomous_drive pure_pursuit_controller
```





session 2
  translation 56.8755 -65.2634 0.400134
  rotation -0.0013009 3.97107e-06 0.999999 -1.0944

session 1
  translation 57.4214 -39.8686 0.400134
  rotation -2.4193992400859864e-06 -0.0007925767510579611 0.9999996859080708 2.0472
