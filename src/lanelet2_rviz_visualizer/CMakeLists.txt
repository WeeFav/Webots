cmake_minimum_required(VERSION 3.14)
project(lanelet2_rviz_visualizer)

# ---- Compiler settings -----------------------------------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

# ---- Find dependencies -----------------------------------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# Lanelet2
find_package(lanelet2_core REQUIRED)
find_package(lanelet2_io REQUIRED)
find_package(lanelet2_projection REQUIRED)

# PROJ
find_package(PkgConfig REQUIRED)
pkg_check_modules(PROJ REQUIRED IMPORTED_TARGET proj)

# ---- Executable ------------------------------------------------------------
add_executable(lanelet2_visualizer
  src/lanelet2_visualizer.cpp
)

target_link_libraries(lanelet2_visualizer PkgConfig::PROJ)

ament_target_dependencies(lanelet2_visualizer
  rclcpp
  visualization_msgs
  geometry_msgs
  std_msgs
  lanelet2_core
  lanelet2_io
  lanelet2_projection
)

# ---- Install ---------------------------------------------------------------
install(TARGETS lanelet2_visualizer
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL          # won't fail if dirs don't exist yet
)

# ---- Ament -----------------------------------------------------------------
ament_package()