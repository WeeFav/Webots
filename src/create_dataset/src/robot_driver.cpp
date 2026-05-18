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
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "create_dataset/robot_driver.hpp"
#include "create_dataset/load_lanes_sumo.hpp"

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
    node = webots_node;
    node->set_parameter(rclcpp::Parameter("use_sim_time", true));

    node->declare_parameter("saving", false);
    node->declare_parameter("data_root", "");
    node->declare_parameter("save_freq", 10);
    node->declare_parameter("save_num", 250);

    saving = node->get_parameter("saving").as_bool();
    data_root = node->get_parameter("data_root").as_string();
    save_freq = node->get_parameter("save_freq").as_int();
    save_num = node->get_parameter("save_num").as_int();

    // camera
    camera_node = wb_supervisor_node_get_from_def("CAMERA");
    camera = wb_robot_get_device("cam0");
    wb_camera_enable(camera, 30);
    
    // camera intrinsic
    width  = wb_camera_get_width(camera);
    height = wb_camera_get_height(camera);
    double fov = wb_camera_get_fov(camera);
    K = get_intrinsic_matrix(width, height, fov);

    vehicle_node = wb_supervisor_node_get_from_def("SUMO_VEHICLE0");

    // ---- Publishers / Subscribers ----


    R_webots_to_opencv <<  0, -1,  0,
                            0,  0, -1,
                            1,  0,  0;
    
    // ---- Lane data ----
    all_lanes_center = load_lane_boundaries("/home/marvin/Webots/src/create_dataset/resource/lanes_sumo.txt");
    seg_vis_color = {{1, cv::Scalar(0, 255, 255)}, {2, cv::Scalar(0, 0, 255)}, {3, cv::Scalar(255, 0, 0)}, {4, cv::Scalar(0, 255, 0)}};
    if (saving) {
        if (std::filesystem::exists(data_root)) {
            throw std::runtime_error("Error: data_root directory already exists: " + data_root);
        }

        // Create /img and /seg subdirectories
        std::filesystem::create_directories(data_root + "/img");
        std::filesystem::create_directories(data_root + "/seg");
    }

    // ZeroMQ
    context = zmq::context_t(1);
    subscriber = zmq::socket_t(context, zmq::socket_type::sub);
    subscriber.connect("tcp://172.21.224.1:5555");
    subscriber.set(zmq::sockopt::subscribe, ""); // Use "" to subscribe to ALL messages
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

    cv::Mat seg(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<bool> lane_exist(4, false);
    lane_detection(seg, lane_exist);

    const double *velocity = wb_supervisor_node_get_velocity(vehicle_node);
    bool zero_vel = (velocity[0] == 0 && velocity[1] == 0 && velocity[2] == 0 && velocity[3] == 0 && velocity[4] == 0 && velocity[5] == 0);
    if (saving && (step_count % save_freq == 0) && save_counter < save_num && !zero_vel) {
        save_lane(bgr, seg, lane_exist);
    }

}

void robot_driver::RobotDriver::lane_detection(cv::Mat &seg, std::vector<bool> &lane_exist) {
    zmq::message_t message;
    auto result = subscriber.recv(message, zmq::recv_flags::dontwait);
    if (result.has_value()) {
        std::string msg(static_cast<char*>(message.data()), message.size());
        // Parse JSON
        std::vector<std::string> lane_ids = nlohmann::json::parse(msg);

        current_lane_id = lane_ids[0];
        left_lane_id = lane_ids[1];
        right_lane_id = lane_ids[2];
        next_lane_id = lane_ids[3];
        next_left_lane_id = lane_ids[4];
        next_right_lane_id = lane_ids[5];
    }

    const double* pos = wb_supervisor_node_get_position(camera_node);
    const double* ori = wb_supervisor_node_get_orientation(camera_node);
    auto extrinsic = get_extrinsic_matrix(pos, ori);

    std::vector<std::pair<Eigen::MatrixXd, int>> all_coords;
    cv::Mat seg_vis(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    
    // current edge
    if (current_lane_id != "None" && all_lanes_center.find(current_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd left_line = all_lanes_center[current_lane_id].left_line;
        Eigen::MatrixXd right_line = all_lanes_center[current_lane_id].right_line;
        
        Eigen::MatrixXd left_coords = world_to_image(left_line, extrinsic);   // N×2
        Eigen::MatrixXd right_coords = world_to_image(right_line, extrinsic);   // N×2

        all_coords.emplace_back(left_coords, 2);
        all_coords.emplace_back(right_coords, 3);
    }
    if (left_lane_id != "None" && all_lanes_center.find(left_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd left_left_line = all_lanes_center[left_lane_id].left_line;
        Eigen::MatrixXd left_left_coords = world_to_image(left_left_line, extrinsic);   // N×2
        all_coords.emplace_back(left_left_coords, 1);
    }
    if (right_lane_id != "None" && all_lanes_center.find(right_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd right_right_line = all_lanes_center[right_lane_id].right_line;
        Eigen::MatrixXd right_right_coords = world_to_image(right_right_line, extrinsic);   // N×2
        all_coords.emplace_back(right_right_coords, 4);
    }

    // next edge
    if (next_lane_id != "None" && all_lanes_center.find(next_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd left_line = all_lanes_center[next_lane_id].left_line;
        Eigen::MatrixXd right_line = all_lanes_center[next_lane_id].right_line;
        
        Eigen::MatrixXd left_coords = world_to_image(left_line, extrinsic);   // N×2
        Eigen::MatrixXd right_coords = world_to_image(right_line, extrinsic);   // N×2

        all_coords.emplace_back(left_coords, 2);
        all_coords.emplace_back(right_coords, 3);
    }
    if (next_left_lane_id != "None" && all_lanes_center.find(next_left_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd left_left_line = all_lanes_center[next_left_lane_id].left_line;
        Eigen::MatrixXd left_left_coords = world_to_image(left_left_line, extrinsic);   // N×2
        all_coords.emplace_back(left_left_coords, 1);
    }
    if (next_right_lane_id != "None" && all_lanes_center.find(next_right_lane_id) != all_lanes_center.end()) {
        Eigen::MatrixXd right_right_line = all_lanes_center[next_right_lane_id].right_line;
        Eigen::MatrixXd right_right_coords = world_to_image(right_right_line, extrinsic);   // N×2
        all_coords.emplace_back(right_right_coords, 4);
    }

    for (const auto& coords_pair : all_coords) {
        Eigen::MatrixXd coords = coords_pair.first;
        int lane_idx = coords_pair.second;
        std::vector<cv::Point> pts;

        for (int i = 0; i < coords.rows(); ++i) {
            int u = static_cast<int>(coords(i, 0));
            int v = static_cast<int>(coords(i, 1));
            if (u >= 0 && u < width && v >= 0 && v < height)
                pts.emplace_back(u, v);
        }

        // for (const auto& pt : pts) {
        //     cv::circle(
        //         img,          // target image
        //         pt,             // center point
        //         5,              // radius
        //         cv::Scalar(0, 0, 255), // BGR color (red)
        //         -1              // negative thickness = filled circle
        //     );
        // }
        
        if (pts.size() >= 2) {
            std::vector<std::vector<cv::Point>> contours = {pts};
            cv::polylines(seg, contours, false, cv::Scalar(lane_idx, lane_idx, lane_idx), 2);
            cv::polylines(seg_vis, contours, false, seg_vis_color[lane_idx], 2);
            lane_exist[lane_idx - 1] = true;
        }
    }

    cv::imshow("segmentation", seg_vis);
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

void robot_driver::RobotDriver::save_lane(cv::Mat &img, cv::Mat &seg, std::vector<bool> &lane_exist) {
    std::string img_filename = std::to_string(save_counter) + ".png";
    std::string img_rel_path = "img/" + img_filename;
    std::string seg_rel_path = "seg/" + img_filename;

    cv::imwrite(data_root + "/" + img_rel_path, img);
    cv::imwrite(data_root + "/" + seg_rel_path, seg);

    // --- Append to train_gt.txt ---
    std::ofstream gt_file(data_root + "/train_gt.txt", std::ios::app);
    if (!gt_file.is_open()) {
        throw std::runtime_error("Error: could not open train_gt.txt for writing");
    }

    gt_file << img_rel_path << " " << seg_rel_path;
    for (bool exists : lane_exist) {
        gt_file << " " << (exists ? 1 : 0);
    }
    gt_file << "\n";

    save_counter++;
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(robot_driver::RobotDriver, webots_ros2_driver::PluginInterface)