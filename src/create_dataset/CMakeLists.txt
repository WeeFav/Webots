cmake_minimum_required(VERSION 3.8)
project(create_dataset)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
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

add_library(${PROJECT_NAME} SHARED
  src/robot_driver.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
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
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)
 
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

add_executable(lane_follower src/lane_follower.cpp)

ament_target_dependencies(lane_follower
  rclcpp
  ackermann_msgs
  sensor_msgs
  std_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  webots_ros2_driver
  pluginlib
)

target_link_libraries(lane_follower
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
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
