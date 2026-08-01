cmake_minimum_required(VERSION 3.8)
project(autonomous_drive)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

if(NOT DEFINED CMAKE_PREFIX_PATH OR NOT CMAKE_PREFIX_PATH)
  set(CMAKE_PREFIX_PATH "/opt/libtorch")
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(webots_ros2_driver REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(webots_ros2_driver REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)
find_package(nlohmann_json REQUIRED)
find_package(Torch REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(nav_msgs REQUIRED)

# Lanelet2
find_package(lanelet2_core REQUIRED)
find_package(lanelet2_io REQUIRED)
find_package(lanelet2_projection REQUIRED)
find_package(lanelet2_routing REQUIRED)
find_package(lanelet2_traffic_rules REQUIRED)

# PROJ
pkg_check_modules(PROJ REQUIRED IMPORTED_TARGET proj)

# PCL
find_package(PCL REQUIRED components common io filters)
find_package(pcl_conversions REQUIRED)


add_library(${PROJECT_NAME} SHARED
  src/robot_driver.cpp
)

add_executable(lane_detection
  src/lane_detection.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ZMQ_INCLUDE_DIRS}
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  ackermann_msgs
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  webots_ros2_driver
  pluginlib
  tf2
  tf2_ros
)

find_library(WEBOTS_CAR_LIB car PATHS "/opt/ros/humble/lib/controller")

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${ZMQ_LIBRARIES}
  nlohmann_json::nlohmann_json
  ${WEBOTS_CAR_LIB}
)

target_link_libraries(lane_detection
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
)

target_include_directories(lane_detection PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
)
 
ament_target_dependencies(lane_detection
  rclcpp
  sensor_msgs
  cv_bridge
)

set_property(TARGET lane_detection PROPERTY CXX_STANDARD 17)

# lanelet2_waypoint_publisher
add_executable(lanelet2_waypoint_publisher
  src/lanelet2_waypoint_publisher.cpp
)

target_link_libraries(lanelet2_waypoint_publisher
  PkgConfig::PROJ
  Eigen3::Eigen
)

target_include_directories(lanelet2_waypoint_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(lanelet2_waypoint_publisher
  rclcpp
  std_msgs
  geometry_msgs
  visualization_msgs
  nav_msgs
  lanelet2_core
  lanelet2_io
  lanelet2_projection
  lanelet2_routing
  lanelet2_traffic_rules
)

set_property(TARGET lanelet2_waypoint_publisher PROPERTY CXX_STANDARD 17)

# pid_controller
add_executable(pid_controller
  src/pid_controller.cpp
)
ament_target_dependencies(pid_controller
  rclcpp
  std_msgs
)
set_property(TARGET pid_controller PROPERTY CXX_STANDARD 17)

# pure_pursuit_controller
add_executable(pure_pursuit_controller
  src/pure_pursuit_controller.cpp
)
target_link_libraries(pure_pursuit_controller
  Eigen3::Eigen
)
target_include_directories(pure_pursuit_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
ament_target_dependencies(pure_pursuit_controller
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  visualization_msgs
)
set_property(TARGET pure_pursuit_controller PROPERTY CXX_STANDARD 17)

# align_pcd_maps
add_executable(align_pcd_maps
  src/align_pcd_maps.cpp
)
target_include_directories(align_pcd_maps PUBLIC
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(align_pcd_maps
  ${PCL_LIBRARIES}
)
set_property(TARGET align_pcd_maps PROPERTY CXX_STANDARD 17)

# pcd_publisher
add_executable(pcd_publisher
  src/pcd_publisher.cpp
)
target_include_directories(pcd_publisher PUBLIC
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(pcd_publisher
  ${PCL_LIBRARIES}
)
ament_target_dependencies(pcd_publisher
  rclcpp
  sensor_msgs
  pcl_conversions
)
set_property(TARGET pcd_publisher PROPERTY CXX_STANDARD 17)

# webots_tf_publisher
add_executable(webots_tf_publisher
  src/webots_tf_publisher.cpp
)
ament_target_dependencies(webots_tf_publisher
  rclcpp
  geometry_msgs
  tf2
  tf2_ros
)
set_property(TARGET webots_tf_publisher PROPERTY CXX_STANDARD 17)


install(DIRECTORY include/
  DESTINATION include
)
 
# Plugin XML (tells pluginlib where to find the driver class)
install(FILES robot_driver.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  resource
  DESTINATION share/${PROJECT_NAME}/
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS lane_detection lanelet2_waypoint_publisher pid_controller pure_pursuit_controller align_pcd_maps pcd_publisher webots_tf_publisher
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/manual_control.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME manual_control
)

pluginlib_export_plugin_description_file(webots_ros2_driver robot_driver.xml)

ament_export_include_directories(
  include
)
ament_export_libraries(
  ${PROJECT_NAME}
)

ament_export_targets(${PROJECT_NAME}Targets)

ament_package()
