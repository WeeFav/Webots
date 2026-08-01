#!/bin/bash
pkill -f ros2
killall -9 rviz
ros2 daemon stop
ros2 daemon start