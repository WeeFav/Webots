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
#include <Eigen/Dense>
#include <Eigen/Geometry>

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

#include "create_dataset/robot_driver.hpp"
#include "create_dataset/load_lanes.hpp"

Eigen::Matrix3d get_intrinsic_matrix(int width, int height, double fov)
{
    double fx = width / (2.0 * std::tan(fov / 2.0));
    double fy = fx;
    double cx = width  / 2.0;
    double cy = height / 2.0;
 
    Eigen::Matrix3d K;
    K << fx,  0, cx,
          0, fy, cy,
          0,  0,  1;
    return K;
}

Eigen::Matrix<double, 3, 4> get_extrinsic_matrix(const double* position,
                                                   const double* orientation)
{
    // Webots gives rotation matrix R such that:  p_world = R * p_local + t
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R(orientation);
    Eigen::Map<const Eigen::Vector3d> t(position);
 
    // World → Camera
    Eigen::Matrix3d R_wc = R.transpose();
    Eigen::Vector3d t_wc = -R_wc * t;
 
    Eigen::Matrix<double, 3, 4> extrinsic;
    extrinsic.leftCols(3)  = R_wc;
    extrinsic.rightCols(1) = t_wc;
    return extrinsic;
}

void robot_driver::RobotDriver::init(webots_ros2_driver::WebotsNode *webots_node, std::unordered_map<std::string, std::string> &parameters) {
    // ---- ROS node ----
    rclcpp::NodeOptions options;
    options.parameter_overrides({rclcpp::Parameter("use_sim_time", true)});    
    node = rclcpp::Node::make_shared("robot_driver_node", options);
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));

    // camera
    camera_node = wb_supervisor_node_get_from_def("CAMERA");
    camera = wb_robot_get_device("cam0");
    wb_camera_enable(camera, 30);
    
    // camera intrinsic
    width  = wb_camera_get_width(camera);
    height = wb_camera_get_height(camera);
    double fov = wb_camera_get_fov(camera);
    K = get_intrinsic_matrix(width, height, fov);

    // ---- Publishers / Subscribers ----


    R_webots_to_opencv <<  0, -1,  0,
                            0,  0, -1,
                            1,  0,  0;
    
    // ---- Lane data ----
    all_lanes_center = load_lanes_txt("/home/marvin/Webots/src/create_dataset/resource/lanes.txt");

    // ZeroMQ
    context = zmq::context_t(1);
    subscriber = zmq::socket_t(context, zmq::socket_type::sub);
    subscriber.connect("tcp://172.21.224.1:5555");
    subscriber.set(zmq::sockopt::subscribe, ""); // Use "" to subscribe to ALL messages
}

// Called every simulation step
void robot_driver::RobotDriver::step() {
    rclcpp::spin_some(node);

    zmq::message_t msg;
    auto result = subscriber.recv(msg, zmq::recv_flags::dontwait);
    if (result.has_value()) {
        std::cout << "Received: " << msg.to_string() << std::endl;
    } else {
        std::cout << "No message yet..." << std::endl;
    }

    const unsigned char *image = wb_camera_get_image(camera);
    
    // Webots format = BGRA (4 channels)
    cv::Mat bgra(height, width, CV_8UC4, (void *)image);

    // Convert to BGR for normal OpenCV usage
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    // Show image
    cv::imshow("Camera", bgr);

    lane_detection();   // uncomment to enable
}

void robot_driver::RobotDriver::lane_detection() {
    const double* pos = wb_supervisor_node_get_position(camera_node);
    const double* ori = wb_supervisor_node_get_orientation(camera_node);
    auto extrinsic = get_extrinsic_matrix(pos, ori);
    Eigen::Vector3d cam_pos(pos[0], pos[1], pos[2]);

    cv::Mat img(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    for (const auto& lane : all_lanes_center) {
        // Quick distance filter on lane centroid
        Eigen::Vector3d center = lane.colwise().mean().transpose();
        if ((center - cam_pos).squaredNorm() > max_lane_dist * max_lane_dist)
            continue;

        auto coords = world_to_image(lane, extrinsic);   // N×2

        std::vector<cv::Point> pts;
        for (int i = 0; i < coords.rows(); ++i) {
            int u = static_cast<int>(coords(i, 0));
            int v = static_cast<int>(coords(i, 1));
            if (u >= 0 && u < width && v >= 0 && v < height)
                pts.emplace_back(u, v);
        }

        for (const auto& pt : pts) {
            cv::circle(
                img,          // target image
                pt,             // center point
                5,              // radius
                cv::Scalar(0, 0, 255), // BGR color (red)
                -1              // negative thickness = filled circle
            );
        }
        
        // if (pts.size() >= 2) {
        //     std::vector<std::vector<cv::Point>> contours = {pts};
        //     cv::polylines(img, contours, false, cv::Scalar(255, 255, 255), 2);
        // }
    }

    cv::imshow("segmentation", img);
    cv::waitKey(1);
}

Eigen::MatrixXd robot_driver::RobotDriver::world_to_image(const Eigen::MatrixXd& points_world, const Eigen::Matrix<double, 3, 4>& extrinsic) {
    // Combined transform: R_webots_to_opencv * extrinsic  →  3×4
    Eigen::Matrix<double, 3, 4> T = R_webots_to_opencv * extrinsic;
    Eigen::Matrix3d R = T.leftCols(3);
    Eigen::Vector3d t = T.rightCols(1);

    // Transform all points: p_cv[i] = R * p_w[i] + t
    Eigen::MatrixXd p_cv = (points_world * R.transpose()).rowwise() + t.transpose(); // N×3

    // Keep only points in front of camera
    std::vector<int> valid_idx;
    for (int i = 0; i < p_cv.rows(); ++i)
        if (p_cv(i, 2) > 1e-6 && p_cv(i, 2) < 70.0)
            valid_idx.push_back(i);

    Eigen::MatrixXd p_valid(valid_idx.size(), 3);
    for (size_t j = 0; j < valid_idx.size(); ++j)
        p_valid.row(j) = p_cv.row(valid_idx[j]);

    if (p_valid.rows() == 0)
        return Eigen::MatrixXd(0, 2);

    // Project: img_hom = (K * p_valid.T).T
    Eigen::MatrixXd img_hom = (K * p_valid.transpose()).transpose(); // M×3

    Eigen::MatrixXd result(img_hom.rows(), 2);
    result.col(0) = img_hom.col(0).array() / img_hom.col(2).array();
    result.col(1) = img_hom.col(1).array() / img_hom.col(2).array();
    return result;
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(robot_driver::RobotDriver, webots_ros2_driver::PluginInterface)