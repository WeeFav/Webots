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
#include <webots/vehicle/car.h>

#include <sstream>
#include <iomanip>
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
    // ---- Webots Driver Init ----
    wbu_driver_init();

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

    // gps
    gps = wb_robot_get_device("gps");
    if (gps != 0) {
        wb_gps_enable(gps, (int)wb_robot_get_basic_time_step());
    }

    vehicle_node = wb_supervisor_node_get_from_def("WEBOTS_VEHICLE0");

    // ---- Static and Dynamic Transform Broadcasters ----
    static_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(node);

    // ---- Publishers / Subscribers ----
    imu_sub = node->create_subscription<sensor_msgs::msg::Imu>(
            "/vehicle/imu", 10,
            std::bind(&RobotDriver::imu_callback, this, std::placeholders::_1));
    ackermann_sub = node->create_subscription<ackermann_msgs::msg::AckermannDrive>(
            "/cmd_ackermann", 10,
            std::bind(&RobotDriver::cmd_ackermann_callback, this, std::placeholders::_1));
    lidar_pub    = node->create_publisher<sensor_msgs::msg::PointCloud2>("/points", 10);
    imu_pub      = node->create_publisher<sensor_msgs::msg::Imu>("/vehicle/imu_interpolated", 10);

    // Manual control inputs
    throttle_sub = node->create_subscription<std_msgs::msg::Float64>(
        "/vehicle/throttle", 10,
        [this](const std_msgs::msg::Float64::SharedPtr msg) {
            if (control_mode_ != "auto") return;
            double throttle = msg->data;
            if (throttle < 0.0) throttle = 0.0;
            if (throttle > 1.0) throttle = 1.0;
            wbu_driver_set_throttle(throttle);
        });

    brake_sub = node->create_subscription<std_msgs::msg::Float64>(
        "/vehicle/brake", 10,
        [this](const std_msgs::msg::Float64::SharedPtr msg) {
            if (control_mode_ != "auto") return;
            double brake = msg->data;
            if (brake < 0.0) brake = 0.0;
            if (brake > 1.0) brake = 1.0;
            wbu_driver_set_brake_intensity(brake);
        });

    steering_sub = node->create_subscription<std_msgs::msg::Float64>(
        "/vehicle/steering_angle", 10,
        [this](const std_msgs::msg::Float64::SharedPtr msg) {
            if (control_mode_ != "auto") return;
            double target_steering = msg->data;
            if (target_steering > 0.5) target_steering = 0.5;
            else if (target_steering < -0.5) target_steering = -0.5;

            // Rate limit steering
            double wheel_angle = target_steering;
            if (wheel_angle - steering_angle > 0.1)
                wheel_angle = steering_angle + 0.1;
            if (wheel_angle - steering_angle < -0.1)
                wheel_angle = steering_angle - 0.1;

            steering_angle = wheel_angle;
            wbu_driver_set_steering_angle(steering_angle);
        });

    // Control mode subscription
    mode_sub = node->create_subscription<std_msgs::msg::String>(
        "/vehicle/control_mode", 10,
        [this](const std_msgs::msg::String::SharedPtr msg) {
            control_mode_ = msg->data;
            RCLCPP_INFO(node->get_logger(), "Control mode switched to: %s", control_mode_.c_str());
            if (control_mode_ == "auto") {
                wbu_driver_set_gear(1);
                wbu_driver_set_cruising_speed(0);
            } else {
                wbu_driver_set_gear(0); // Neutral
            }
        });

    // Vehicle state feedback
    velocity_pub = node->create_publisher<std_msgs::msg::Float64>("/vehicle/current_velocity", 10);
    pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("/vehicle/world_pose", 10);
    lidar_pose_pub = node->create_publisher<geometry_msgs::msg::PoseStamped>("/lidar/world_pose", 10);
    target_speed_pub = node->create_publisher<std_msgs::msg::Float64>("/vehicle/target_speed", 10);

    if (!node->has_parameter("target_speed")) {
        node->declare_parameter<double>("target_speed", 5.0);
    }

    auto check_bool_param = [this](const std::string& name) -> bool {
        if (!node->has_parameter(name)) {
            try {
                node->declare_parameter<bool>(name, false);
            } catch (...) {
                try {
                    node->declare_parameter<std::string>(name, "false");
                } catch (...) {}
            }
        }
        try {
            auto p = node->get_parameter(name);
            if (p.get_type() == rclcpp::PARAMETER_BOOL) {
                return p.as_bool();
            } else if (p.get_type() == rclcpp::PARAMETER_STRING) {
                std::string s = p.as_string();
                return (s == "true" || s == "True" || s == "1");
            }
        } catch (...) {}
        return false;
    };

    if (check_bool_param("print_session")) {
        print_session_ = true;
    }
    if (parameters.find("print_session") != parameters.end() &&
        (parameters["print_session"] == "true" || parameters["print_session"] == "1")) {
        print_session_ = true;
    }

    if (print_session_) {
        RCLCPP_INFO(node->get_logger(), "Print session info enabled (skipping first %d steps, collecting %d samples).",
                    print_step_skip_, samples_to_collect_);
    }

    RCLCPP_INFO(node->get_logger(), "RobotDriver initialized.");
    RCLCPP_INFO(node->get_logger(), "Wheelbase: %f", wbu_car_get_wheelbase());
    
}

void autonomous_drive::RobotDriver::step() {
    wbu_driver_step();
    rclcpp::spin_some(node->get_node_base_interface());
    step_count++;

    // const double* gps_coords = wb_gps_get_values(gps);
    // RCLCPP_INFO(node->get_logger(), "gps_coords: %f, %f, %f", gps_coords[0], gps_coords[1], gps_coords[2]);

    if (print_session_ && !session_printed_) {
        if (step_count > print_step_skip_) {
            if (gps_samples_.size() < static_cast<size_t>(samples_to_collect_)) {
                if (gps != 0) {
                    const double* coords = wb_gps_get_values(gps);
                    if (coords != nullptr) {
                        gps_samples_.push_back({coords[0], coords[1], coords[2]});
                    }
                }
            }

            if ((gps == 0 || gps_samples_.size() >= static_cast<size_t>(samples_to_collect_)) &&
                imu_samples_.size() >= static_cast<size_t>(samples_to_collect_)) {
                print_session_info();
                session_printed_ = true;
            } else if (step_count > (print_step_skip_ + samples_to_collect_ + 50) &&
                       (!gps_samples_.empty() || !imu_samples_.empty())) {
                print_session_info();
                session_printed_ = true;
            }
        }
    }

    // Automatic transmission logic in autonomous mode
    if (control_mode_ == "auto") {
        int current_gear = wbu_driver_get_gear();
        double rpm = wbu_driver_get_rpm();
        int max_gear = wbu_driver_get_gear_number();
        if (current_gear <= 0) {
            wbu_driver_set_gear(1);
        } else {
            if (rpm > 4500.0 && current_gear < max_gear) {
                wbu_driver_set_gear(current_gear + 1);
            } else if (rpm < 1500.0 && current_gear > 1) {
                wbu_driver_set_gear(current_gear - 1);
            }
        }
    }

    if (step_count % 3 == 0) {
        publish_lidar();
    }

    // Publish current velocity
    if (velocity_pub != nullptr) {
        std_msgs::msg::Float64 speed_msg;
        if (gps != 0) {
            speed_msg.data = wb_gps_get_speed(gps); // wb_gps_get_speed already returns m/s
        } else {
            speed_msg.data = wbu_driver_get_current_speed() / 3.6; // fallback if no gps
        }
        velocity_pub->publish(speed_msg);
    }

    // Publish target speed
    if (target_speed_pub != nullptr) {
        std_msgs::msg::Float64 target_msg;
        double target_speed = 20.0 / 3.6;
        // node->get_parameter("target_speed", target_speed);
        target_msg.data = target_speed;
        target_speed_pub->publish(target_msg);
    }

    // // Publish world pose and TF
    publish_pose();

    if (!transform_published && lidar_node != nullptr) {
        const double* pos = wb_supervisor_node_get_position(lidar_node);
        const double* rot = wb_supervisor_node_get_orientation(lidar_node);
        if (pos != nullptr && rot != nullptr) {
            tf2::Matrix3x3 m(
                rot[0], rot[1], rot[2],
                rot[3], rot[4], rot[5],
                rot[6], rot[7], rot[8]
            );
            tf2::Quaternion q;
            m.getRotation(q);

            tf2::Quaternion q_offset;
            q_offset.setRPY(0.0, 0.0, 0.0);
            tf2::Quaternion q_final = q * q_offset;
            q_final.normalize();

            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header.stamp = node->get_clock()->now();
            tf_msg.header.frame_id = "map";
            tf_msg.child_frame_id = "inital_lidar";

            tf_msg.transform.translation.x = pos[0];
            tf_msg.transform.translation.y = pos[1];
            tf_msg.transform.translation.z = pos[2];

            tf_msg.transform.rotation.x = q_final.x();
            tf_msg.transform.rotation.y = q_final.y();
            tf_msg.transform.rotation.z = q_final.z();
            tf_msg.transform.rotation.w = q_final.w();

            static_broadcaster->sendTransform(tf_msg);
            transform_published = true;
            RCLCPP_INFO(node->get_logger(), "Published static transform map -> lio_map: Translation (%.2f, %.2f, %.2f)", 
                        pos[0], pos[1], pos[2]);
        }
    }

    // const unsigned char *image = wb_camera_get_image(camera);
    // if (image != nullptr) {
    //     // Webots format = BGRA (4 channels)
    //     cv::Mat bgra(height, width, CV_8UC4, (void *)image);

    //     // Convert to BGR for normal OpenCV usage
    //     cv::Mat bgr;
    //     cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

    //     // Show image
    //     cv::imshow("Camera", bgr);
    //     cv::waitKey(1);
    // }
}

void autonomous_drive::RobotDriver::publish_pose() {
    auto now = node->get_clock()->now();

    // -------- Publish vehicle_node pose & TF --------
    if (vehicle_node != nullptr) {
        const double* pos = wb_supervisor_node_get_position(vehicle_node);
        const double* rot = wb_supervisor_node_get_orientation(vehicle_node);
        if (pos != nullptr && rot != nullptr) {
            tf2::Matrix3x3 m(
                rot[0], rot[1], rot[2],
                rot[3], rot[4], rot[5],
                rot[6], rot[7], rot[8]);

            tf2::Quaternion q;
            m.getRotation(q);

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = now;
            pose_msg.header.frame_id = "map";

            pose_msg.pose.position.x = pos[0];
            pose_msg.pose.position.y = pos[1];
            pose_msg.pose.position.z = pos[2];

            pose_msg.pose.orientation.x = q.x();
            pose_msg.pose.orientation.y = q.y();
            pose_msg.pose.orientation.z = q.z();
            pose_msg.pose.orientation.w = q.w();

            if (pose_pub != nullptr) {
                pose_pub->publish(pose_msg);
            }

            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header = pose_msg.header;
            tf_msg.child_frame_id = "gt_vehicle";

            tf_msg.transform.translation.x = pos[0];
            tf_msg.transform.translation.y = pos[1];
            tf_msg.transform.translation.z = pos[2];

            tf_msg.transform.rotation = pose_msg.pose.orientation;

            if (tf_broadcaster != nullptr) {
                tf_broadcaster->sendTransform(tf_msg);
            }
        }
    }

    // -------- Publish lidar_node pose & TF --------
    if (lidar_node != nullptr) {
        const double* pos = wb_supervisor_node_get_position(lidar_node);
        const double* rot = wb_supervisor_node_get_orientation(lidar_node);
        if (pos != nullptr && rot != nullptr) {
            tf2::Matrix3x3 m(
                rot[0], rot[1], rot[2],
                rot[3], rot[4], rot[5],
                rot[6], rot[7], rot[8]);

            tf2::Quaternion q;
            m.getRotation(q);

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp = now;
            pose_msg.header.frame_id = "map";

            pose_msg.pose.position.x = pos[0];
            pose_msg.pose.position.y = pos[1];
            pose_msg.pose.position.z = pos[2];

            pose_msg.pose.orientation.x = q.x();
            pose_msg.pose.orientation.y = q.y();
            pose_msg.pose.orientation.z = q.z();
            pose_msg.pose.orientation.w = q.w();

            if (lidar_pose_pub != nullptr) {
                lidar_pose_pub->publish(pose_msg);
            }

            geometry_msgs::msg::TransformStamped tf_msg;
            tf_msg.header = pose_msg.header;
            tf_msg.child_frame_id = "gt_lidar";

            tf_msg.transform.translation.x = pos[0];
            tf_msg.transform.translation.y = pos[1];
            tf_msg.transform.translation.z = pos[2];

            tf_msg.transform.rotation = pose_msg.pose.orientation;

            if (tf_broadcaster != nullptr) {
                tf_broadcaster->sendTransform(tf_msg);
            }
        }
    }
}

void autonomous_drive::RobotDriver::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
{
    if (print_session_ && !session_printed_) {
        if (step_count > print_step_skip_ && imu_samples_.size() < static_cast<size_t>(samples_to_collect_)) {
            imu_samples_.push_back({msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w});
        }
    }

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
    msg.header.frame_id = "gt_lidar";
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

void autonomous_drive::RobotDriver::cmd_ackermann_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg) {
    if (control_mode_ != "manual") return;
    double target_speed = msg->speed * 3.6; // convert m/s to km/h
    double target_steering = msg->steering_angle; // radians
    
    // RCLCPP_INFO(node->get_logger(), "cmd_ackermann callback: speed_msg = %.3f m/s, target_speed = %.3f km/h, current_speed = %.3f", msg->speed, target_speed, wbu_driver_get_current_speed());

    // Range checking
    if (target_speed > 250.0) target_speed = 250.0;
    if (target_speed < -250.0) target_speed = -250.0;

    if (target_steering > 0.5) target_steering = 0.5;
    else if (target_steering < -0.5) target_steering = -0.5;

    // Rate limiting steering to match autonomous_vehicle.cpp
    double wheel_angle = target_steering;
    if (wheel_angle - steering_angle > 0.1)
        wheel_angle = steering_angle + 0.1;
    if (wheel_angle - steering_angle < -0.1)
        wheel_angle = steering_angle - 0.1;

    steering_angle = wheel_angle;

    // Set gear and cruising speed for manual control
    if (target_speed > 0.0) {
        wbu_driver_set_gear(1);
        wbu_driver_set_cruising_speed(target_speed);
    } else if (target_speed < 0.0) {
        wbu_driver_set_gear(-1);
        wbu_driver_set_cruising_speed(-target_speed);
    } else {
        wbu_driver_set_cruising_speed(0.0);
    }
    wbu_driver_set_cruising_speed(target_speed);
    wbu_driver_set_steering_angle(steering_angle);
}

void autonomous_drive::RobotDriver::print_session_info() {
    std::array<double, 3> veh_pos = {0.0, 0.0, 0.0};
    std::array<double, 4> veh_rot = {0.0, 0.0, 1.0, 0.0};

    if (vehicle_node != nullptr) {
        const double* pos = wb_supervisor_node_get_position(vehicle_node);
        if (pos != nullptr) {
            veh_pos = {pos[0], pos[1], pos[2]};
        }

        WbFieldRef rot_field = wb_supervisor_node_get_field(vehicle_node, "rotation");
        if (rot_field != nullptr) {
            const double* r = wb_supervisor_field_get_sf_rotation(rot_field);
            if (r != nullptr) {
                veh_rot = {r[0], r[1], r[2], r[3]};
            }
        } else {
            const double* rot_mat = wb_supervisor_node_get_orientation(vehicle_node);
            if (rot_mat != nullptr) {
                Eigen::Matrix3d R;
                R << rot_mat[0], rot_mat[1], rot_mat[2],
                     rot_mat[3], rot_mat[4], rot_mat[5],
                     rot_mat[6], rot_mat[7], rot_mat[8];
                Eigen::AngleAxisd aa(R);
                veh_rot = {aa.axis().x(), aa.axis().y(), aa.axis().z(), aa.angle()};
            }
        }
    }

    std::array<double, 3> lidar_pos = {0.0, 0.0, 0.0};
    std::array<double, 4> lidar_rot = {0.0, 0.0, 1.0, 0.0};

    if (lidar_node != nullptr) {
        const double* pos = wb_supervisor_node_get_position(lidar_node);
        if (pos != nullptr) {
            lidar_pos = {pos[0], pos[1], pos[2]};
        }

        const double* rot_mat = wb_supervisor_node_get_orientation(lidar_node);
        if (rot_mat != nullptr) {
            Eigen::Matrix3d R;
            R << rot_mat[0], rot_mat[1], rot_mat[2],
                 rot_mat[3], rot_mat[4], rot_mat[5],
                 rot_mat[6], rot_mat[7], rot_mat[8];
            Eigen::AngleAxisd aa(R);
            lidar_rot = {aa.axis().x(), aa.axis().y(), aa.axis().z(), aa.angle()};
        }
    }

    double avg_gps[3] = {0.0, 0.0, 0.0};
    if (!gps_samples_.empty()) {
        for (const auto& s : gps_samples_) {
            avg_gps[0] += s[0];
            avg_gps[1] += s[1];
            avg_gps[2] += s[2];
        }
        avg_gps[0] /= gps_samples_.size();
        avg_gps[1] /= gps_samples_.size();
        avg_gps[2] /= gps_samples_.size();
    }

    double avg_imu[4] = {0.0, 0.0, 0.0, 0.0};
    if (!imu_samples_.empty()) {
        for (const auto& s : imu_samples_) {
            avg_imu[0] += s[0];
            avg_imu[1] += s[1];
            avg_imu[2] += s[2];
            avg_imu[3] += s[3];
        }
        avg_imu[0] /= imu_samples_.size();
        avg_imu[1] /= imu_samples_.size();
        avg_imu[2] /= imu_samples_.size();
        avg_imu[3] /= imu_samples_.size();
    }

    std::ostringstream ss;
    ss << std::setprecision(16);
    ss << "  vehicle translation: " << veh_pos[0] << " " << veh_pos[1] << " " << veh_pos[2] << "\n";
    ss << "  vehicle rotation: " << veh_rot[0] << " " << veh_rot[1] << " " << veh_rot[2] << " " << veh_rot[3] << "\n";
    ss << std::fixed << std::setprecision(6);
    ss << "  vehicle GPS: " << avg_gps[0] << ", " << avg_gps[1] << ", " << avg_gps[2] << " \n";
    ss << std::defaultfloat << std::setprecision(16);
    ss << "  vehicle IMU: " << avg_imu[0] << " " << avg_imu[1] << " " << avg_imu[2] << " " << avg_imu[3] << "\n";
    ss << "  lidar translation: " << lidar_pos[0] << " " << lidar_pos[1] << " " << lidar_pos[2] << "\n";
    ss << "  lidar rotation: " << lidar_rot[0] << " " << lidar_rot[1] << " " << lidar_rot[2] << " " << lidar_rot[3];

    RCLCPP_INFO(node->get_logger(), "\n%s", ss.str().c_str());
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(autonomous_drive::RobotDriver, webots_ros2_driver::PluginInterface)