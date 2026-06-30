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

static constexpr double IMU_STEP_DT_SEC   = 0.033; // basicTimeStep
static constexpr double IMU_OUT_DT_SEC    = 0.002; // desired IMU step
static constexpr int    IMU_INTERP_STEPS  = IMU_STEP_DT_SEC / IMU_OUT_DT_SEC;    // sub-samples per Webots step

void autonomous_drive::RobotDriver::init(webots_ros2_driver::WebotsNode *webots_node, std::unordered_map<std::string, std::string> &parameters) {
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

    // lidar
    lidar_node = wb_supervisor_node_get_from_def("LIDAR");
    lidar = wb_robot_get_device("lidar");
    wb_lidar_enable(lidar, 99);
    wb_lidar_enable_point_cloud(lidar);

    // imu
    inertial_unit  = wb_robot_get_device("inertial_unit");
    accelerometer  = wb_robot_get_device("accelerometer");
    gyro           = wb_robot_get_device("gyro");
    wb_inertial_unit_enable(inertial_unit,  2);
    wb_accelerometer_enable(accelerometer,  2);
    wb_gyro_enable(gyro,                    2);

    vehicle_node = wb_supervisor_node_get_from_def("SUMO_VEHICLE0");

    // ---- Publishers / Subscribers ----
    imu_sub = node->create_subscription<sensor_msgs::msg::Imu>(
            "/vehicle/imu", 10,
            std::bind(&RobotDriver::imu_callback, this, std::placeholders::_1));
    lidar_pub    = node->create_publisher<sensor_msgs::msg::PointCloud2>("/points", 10);
    imu_pub      = node->create_publisher<sensor_msgs::msg::Imu>("/vehicle/imu_interpolated", 10);
    
}

// Called every simulation step
void autonomous_drive::RobotDriver::step() {
    rclcpp::spin_some(node->get_node_base_interface());
    step_count++;
    if (step_count % 3 == 0) {
        publish_lidar();
    }

    const unsigned char *image = wb_camera_get_image(camera);
    
    // Webots format = BGRA (4 channels)
    cv::Mat bgra(height, width, CV_8UC4, (void *)image);

    // Convert to BGR for normal OpenCV usage
    cv::Mat bgr;
    cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    // Show image
    cv::imshow("Camera", bgr);
}

void autonomous_drive::RobotDriver::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    if (!imu_prev_valid_) {
        // First message: store as both previous and current; nothing to interpolate yet.
        imu_prev_ = *msg;
        imu_prev_valid_ = true;
        return;
    }
 
    const sensor_msgs::msg::Imu& p = imu_prev_;   // previous sample
    const sensor_msgs::msg::Imu& c = *msg;        // current sample
 
    // Base timestamp: previous sample's stamp (start of the interval)
    const rclcpp::Time t0(p.header.stamp);
 
    for (int i = 0; i < IMU_INTERP_STEPS; ++i) {
        const double alpha = static_cast<double>(i) / IMU_INTERP_STEPS;
 
        sensor_msgs::msg::Imu out;
        out.header.frame_id = c.header.frame_id;
        out.header.stamp    = t0 + rclcpp::Duration::from_seconds(i * IMU_OUT_DT_SEC);
 
        // --- Linear interpolation of linear acceleration ---
        out.linear_acceleration.x = p.linear_acceleration.x + alpha * (c.linear_acceleration.x - p.linear_acceleration.x);
        out.linear_acceleration.y = p.linear_acceleration.y + alpha * (c.linear_acceleration.y - p.linear_acceleration.y);
        out.linear_acceleration.z = p.linear_acceleration.z + alpha * (c.linear_acceleration.z - p.linear_acceleration.z);
 
        // --- Linear interpolation of angular velocity ---
        out.angular_velocity.x = p.angular_velocity.x + alpha * (c.angular_velocity.x - p.angular_velocity.x);
        out.angular_velocity.y = p.angular_velocity.y + alpha * (c.angular_velocity.y - p.angular_velocity.y);
        out.angular_velocity.z = p.angular_velocity.z + alpha * (c.angular_velocity.z - p.angular_velocity.z);
 
        // --- SLERP for orientation quaternion ---
        Eigen::Quaterniond q0(p.orientation.w, p.orientation.x, p.orientation.y, p.orientation.z);
        Eigen::Quaterniond q1(c.orientation.w, c.orientation.x, c.orientation.y, c.orientation.z);
        Eigen::Quaterniond qi = q0.slerp(alpha, q1);
 
        out.orientation.w = qi.w();
        out.orientation.x = qi.x();
        out.orientation.y = qi.y();
        out.orientation.z = qi.z();
 
        // Propagate covariances from the current message (conservative)
        // out.orientation_covariance         = c.orientation_covariance;
        // out.angular_velocity_covariance    = c.angular_velocity_covariance;
        // out.linear_acceleration_covariance = c.linear_acceleration_covariance;

        out.orientation_covariance = {
            1e-3, 0, 0,
            0, 1e-3, 0,
            0, 0, 1e-3
        };

        out.angular_velocity_covariance = {
            1e-4, 0, 0,
            0, 1e-4, 0,
            0, 0, 1e-4
        };

        out.linear_acceleration_covariance = {
            1e-2, 0, 0,
            0, 1e-2, 0,
            0, 0, 1e-2
        };

        imu_pub->publish(out);
    }
 
    // Slide window: current becomes previous for the next step
    imu_prev_ = *msg;
}

void autonomous_drive::RobotDriver::publish_lidar() {
    // Pack Webots LiDAR point cloud into a PointCloud2 message.
    const WbLidarPoint* pts = wb_lidar_get_point_cloud(lidar);
    if (pts == NULL) return;
    int num_points = wb_lidar_get_number_of_points(lidar);
    sensor_msgs::msg::PointCloud2 msg;
    msg.header.stamp    = node->get_clock()->now();
    msg.header.frame_id = "velodyne";
    msg.height    = 1;
    msg.width     = static_cast<uint32_t>(num_points);
    msg.is_bigendian = false;
    msg.is_dense  = true;
    // Fields: x(f32), y(f32), z(f32), ring(u16), time(f32)
    sensor_msgs::msg::PointField pf;
    pf.name     = "x"; pf.offset = 0;  pf.datatype = sensor_msgs::msg::PointField::FLOAT32; pf.count = 1;
    msg.fields.push_back(pf);
    pf.name     = "y"; pf.offset = 4;
    msg.fields.push_back(pf);
    pf.name     = "z"; pf.offset = 8;
    msg.fields.push_back(pf);
    pf.name     = "intensity"; pf.offset = 12;
    msg.fields.push_back(pf);
    pf.name     = "ring"; pf.offset = 16; pf.datatype = sensor_msgs::msg::PointField::UINT16;
    msg.fields.push_back(pf);
    pf.name     = "time"; pf.offset = 18; pf.datatype = sensor_msgs::msg::PointField::FLOAT32;
    msg.fields.push_back(pf);

    msg.point_step = 22;
    msg.row_step   = msg.point_step * msg.width;
    msg.data.resize(msg.row_step);

    float start_time = pts[num_points - 1].time; 

    for (int i = num_points - 1; i >= 0; i--) {
        uint8_t* base = msg.data.data() + i * msg.point_step;
        float x = static_cast<float>(pts[i].x);
        float y = static_cast<float>(pts[i].y);
        float z = static_cast<float>(pts[i].z);
        float intensity = 0.0;
        uint16_t layer = static_cast<uint16_t>(pts[i].layer_id);
        float    t_pt  = static_cast<float>(pts[i].time) - start_time;
        std::memcpy(base +  0, &x,     4);
        std::memcpy(base +  4, &y,     4);
        std::memcpy(base +  8, &z,     4);
        std::memcpy(base +  12, &intensity,     4);
        std::memcpy(base + 16, &layer, 2);
        std::memcpy(base + 18, &t_pt,  4);
    }
    lidar_pub->publish(msg);
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(autonomous_drive::RobotDriver, webots_ros2_driver::PluginInterface)