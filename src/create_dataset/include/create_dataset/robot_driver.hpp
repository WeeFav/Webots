#pragma once
 
#include "rclcpp/macros.hpp"
#include "webots_ros2_driver/PluginInterface.hpp"
#include "webots_ros2_driver/WebotsNode.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"

#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include <webots/robot.h>
#include <webots/camera.h>
#include <webots/lidar.h>
#include <webots/inertial_unit.h>
#include <webots/accelerometer.h>
#include <webots/gyro.h>
#include <webots/supervisor.h>   // WbNodeRef, WbFieldRef, wb_supervisor_*

using Corners8x3 = Eigen::Matrix<double, 8, 3, Eigen::RowMajor>; // 8 corners × (x,y,z)
 
struct BoxInfo {
    Eigen::Matrix3d R_local_to_vehicle;
    Eigen::Vector3d t_local_to_vehicle;
    Eigen::Vector3d size;
};
 
struct VehicleInfo {
    WbNodeRef node;
    Corners8x3 corners_vehicle; // corners in vehicle frame
};

namespace robot_driver {
class RobotDriver : public webots_ros2_driver::PluginInterface {
public:
    void step() override;
    void init(webots_ros2_driver::WebotsNode *node,
        std::unordered_map<std::string, std::string> &parameters) override;

private:
    void cmd_ackermann_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg);
    void publish_lidar();
    void lane_detection();
    Eigen::MatrixXd world_to_image(const Eigen::MatrixXd& points_world, const Eigen::Matrix<double, 3, 4>& extrinsic);
    void object_detection();
    std::vector<BoxInfo> extract_boxes(
        WbNodeRef node,
        const Eigen::Matrix3d& parent_R,
        const Eigen::Vector3d& parent_t);
    Corners8x3 get_box_corners(const BoxInfo& box);
    Corners8x3 get_bounding_box(const std::vector<BoxInfo>& boxes);
    visualization_msgs::msg::MarkerArray corners_to_marker_array(const std::vector<Corners8x3>& all_corners);

    // ---- ROS / Webots handles ----
    rclcpp::Node::SharedPtr   node;

    WbDeviceTag camera;
    WbDeviceTag lidar;
    WbDeviceTag inertial_unit;
    WbDeviceTag accelerometer;
    WbDeviceTag gyro;
    WbNodeRef camera_node;
    WbNodeRef lidar_node;

    rclcpp::Subscription<ackermann_msgs::msg::AckermannDrive>::SharedPtr ackermann_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                lane_seg_pub;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr   obj_det_pub;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr          lidar_pub;
 
    // ---- Camera intrinsics / extrinsics ----
    Eigen::Matrix3d K;
    Eigen::Matrix3d R_webots_to_opencv;
    int width  = 0;
    int height = 0;
 
    // ---- Lanes ----
    // Each lane is a list of 3-D points (N×3).
    std::vector<Eigen::MatrixX3d> all_lanes_center;
    double max_lane_dist = 50.0;
 
    // ---- Vehicles ----
    std::unordered_map<int, VehicleInfo> vehicles;
    double max_obj_dist = 50.0;

    long step_count = 0;
};
} // namespace robot_driver