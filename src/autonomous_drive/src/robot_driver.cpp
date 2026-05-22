#include <rclcpp/rclcpp.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/header.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/lidar.h>
#include <webots/inertial_unit.h>
#include <webots/accelerometer.h>
#include <webots/gyro.h>
#include <webots/supervisor.h>   // wb_supervisor_node_get_from_def, wb_supervisor_field_*, etc.
#include <webots/vehicle/driver.h>

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "autonomous_drive/robot_driver.hpp"

void robot_driver::RobotDriver::init(webots_ros2_driver::WebotsNode *webots_node, std::unordered_map<std::string, std::string> &parameters) {
    // ---- ROS node ----
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("use_sim_time", true)});    
    node = webots_node;
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));

    // camera
    camera_node = wb_supervisor_node_get_from_def("CAMERA");
    camera = wb_robot_get_device("cam0");
    wb_camera_enable(camera, 30);
    
    // camera intrinsic
    width  = wb_camera_get_width(camera);
    height = wb_camera_get_height(camera);
    double fov = wb_camera_get_fov(camera);

    vehicle_node = wb_supervisor_node_get_from_def("SUMO_VEHICLE0");

    // ---- Publishers / Subscribers ----


    R_webots_to_opencv <<  0, -1,  0,
                            0,  0, -1,
                            1,  0,  0;
    
    // ---- Lane data ----
    seg_vis_color = {{1, cv::Scalar(0, 255, 255)}, {2, cv::Scalar(0, 0, 255)}, {3, cv::Scalar(255, 0, 0)}, {4, cv::Scalar(0, 255, 0)}};
}

// Called every simulation step
void robot_driver::RobotDriver::step() {
    rclcpp::spin_some(node->get_node_base_interface());
    step_count++;

    const unsigned char *image = wb_camera_get_image(camera);
    
    // Webots format = BGRA (4 channels)
    cv::Mat bgra(height, width, CV_8UC4, (void *)image);

    // Convert to BGR for normal OpenCV usage
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    // Show image
    cv::imshow("Camera", bgr);
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(robot_driver::RobotDriver, webots_ros2_driver::PluginInterface)