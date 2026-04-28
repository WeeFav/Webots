#!/bin/bash
pkill -f ros2
ros2 daemon stop
ros2 daemon start