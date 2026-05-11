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

static constexpr double IMU_STEP_DT_SEC   = 0.033; // basicTimeStep
static constexpr double IMU_OUT_DT_SEC    = 0.002; // desired IMU step
static constexpr int    IMU_INTERP_STEPS  = IMU_STEP_DT_SEC / IMU_OUT_DT_SEC;    // sub-samples per Webots step

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
    // rclcpp::init(0, nullptr);
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

    // ---- Publishers / Subscribers ----
    ackermann_sub = node->create_subscription<ackermann_msgs::msg::AckermannDrive>(
        "cmd_ackermann", 1,
        std::bind(&RobotDriver::cmd_ackermann_callback, this, std::placeholders::_1));
    imu_sub = node->create_subscription<sensor_msgs::msg::Imu>(
            "/vehicle/imu", 10,
            std::bind(&RobotDriver::imu_callback, this, std::placeholders::_1));

    lane_seg_pub = node->create_publisher<sensor_msgs::msg::Image>("/segmentation", 10);
    obj_det_pub  = node->create_publisher<visualization_msgs::msg::MarkerArray>("/bbox_markers", 10);
    lidar_pub    = node->create_publisher<sensor_msgs::msg::PointCloud2>("/points", 10);
    imu_pub      = node->create_publisher<sensor_msgs::msg::Imu>("/vehicle/imu_interpolated", 10);
    
    R_webots_to_opencv <<  0, -1,  0,
                            0,  0, -1,
                            1,  0,  0;
    
    // ---- Lane data ----
    // all_lanes_center = load_lanes_txt("/path/to/lanes.txt");

    // ---- Vehicles ----
    for (int i = 0; i < 100; ++i) {
        std::string def_name = "SUMO_VEHICLE" + std::to_string(i);
        WbNodeRef veh_node = wb_supervisor_node_get_from_def(def_name.c_str());

        if (veh_node == NULL) break;

        WbFieldRef bounding_field = wb_supervisor_node_get_base_node_field(veh_node, "boundingObject");
        if (bounding_field == NULL) break;

        WbNodeRef bounding_node = wb_supervisor_field_get_sf_node(bounding_field);

        std::vector<BoxInfo> boxes = extract_boxes(bounding_node,
                                                    Eigen::Matrix3d::Identity(),
                                                    Eigen::Vector3d::Zero());
        Corners8x3 corners = get_bounding_box(boxes);
        vehicles[i] = {veh_node, corners};
    } 

}

// Called every simulation step
void robot_driver::RobotDriver::step() {
    rclcpp::spin_some(node);
    step_count++;
    if (step_count % 3 == 0) {
        publish_lidar();
    }
    // lane_detection();   // uncomment to enable
    object_detection();
}

void robot_driver::RobotDriver::cmd_ackermann_callback(const ackermann_msgs::msg::AckermannDrive::SharedPtr msg) {
    if (step_count > 150) {
        wbu_driver_set_cruising_speed(msg->speed);
        wbu_driver_set_steering_angle(msg->steering_angle);
    }
}

void robot_driver::RobotDriver::imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg)
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

void robot_driver::RobotDriver::publish_lidar() {
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

        if (pts.size() >= 2) {
            std::vector<std::vector<cv::Point>> contours = {pts};
            cv::polylines(img, contours, false, cv::Scalar(255, 255, 255), 2);
        }
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
        if (p_cv(i, 2) > 1e-6 && p_cv(i, 2) < 50.0)
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

void robot_driver::RobotDriver::object_detection() {
    const double* lidar_pos_raw = wb_supervisor_node_get_position(lidar_node);
    const double* lidar_ori_raw = wb_supervisor_node_get_orientation(lidar_node);

    Eigen::Vector3d t_lidar_world(lidar_pos_raw[0], lidar_pos_raw[1], lidar_pos_raw[2]);
    Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_lidar_world(lidar_ori_raw);

    std::vector<Corners8x3> all_corners;

    for (auto& [id, veh] : vehicles) {
        const double* vp = wb_supervisor_node_get_position(veh.node);
        Eigen::Vector3d t_veh_world(vp[0], vp[1], vp[2]);

        if ((t_veh_world - t_lidar_world).squaredNorm() > max_obj_dist * max_obj_dist)
            continue;

        const double* vo = wb_supervisor_node_get_orientation(veh.node);
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> R_veh_world(vo);

        Eigen::Matrix3d R_combined = R_lidar_world.transpose() * R_veh_world;
        Eigen::Vector3d t_combined = R_lidar_world.transpose() * (t_veh_world - t_lidar_world);

        // corners_lidar = (corners_vehicle @ R_combined.T) + t_combined
        Corners8x3 corners_lidar =
            (veh.corners_vehicle * R_combined.transpose()).rowwise() + t_combined.transpose();
        all_corners.push_back(corners_lidar);
    }

    auto marker_array = corners_to_marker_array(all_corners);
    obj_det_pub->publish(marker_array);
}

/**
 * Recursively walk a Webots bounding-object tree and collect Box primitives.
 */
std::vector<BoxInfo> robot_driver::RobotDriver::extract_boxes(WbNodeRef node,
                                    const Eigen::Matrix3d& parent_R,
                                    const Eigen::Vector3d& parent_t) {
    std::vector<BoxInfo> boxes;
    if (!node) return boxes;

    std::string node_type = wb_supervisor_node_get_type_name(node);

    if (node_type == "Group") {
        WbFieldRef children = wb_supervisor_node_get_field(node, "children");
        int count = wb_supervisor_field_get_count(children);
        for (int i = 0; i < count; ++i) {
            WbNodeRef child = wb_supervisor_field_get_mf_node(children, i);
            for (auto& b : extract_boxes(child, parent_R, parent_t))
                boxes.push_back(b);
        }

    } else if (node_type == "Pose") {
        WbFieldRef trans_field = wb_supervisor_node_get_field(node, "translation");
        WbFieldRef rot_field   = wb_supervisor_node_get_field(node, "rotation");

        const double* tv = wb_supervisor_field_get_sf_vec3f(trans_field);
        const double* rv = wb_supervisor_field_get_sf_rotation(rot_field);

        Eigen::Vector3d axis(rv[0], rv[1], rv[2]);
        double angle = rv[3];
        Eigen::AngleAxisd aa(angle, axis.normalized());
        Eigen::Matrix3d R = aa.toRotationMatrix();

        Eigen::Matrix3d new_R = parent_R * R;
        Eigen::Vector3d new_t = parent_R * Eigen::Vector3d(tv[0], tv[1], tv[2]) + parent_t;

        WbFieldRef children = wb_supervisor_node_get_field(node, "children");
        int count = wb_supervisor_field_get_count(children);
        for (int i = 0; i < count; ++i) {
            WbNodeRef child = wb_supervisor_field_get_mf_node(children, i);
            for (auto& b : extract_boxes(child, new_R, new_t))
                boxes.push_back(b);
        }

    } else if (node_type == "Box") {
        WbFieldRef size_field = wb_supervisor_node_get_field(node, "size");
        const double* sv = wb_supervisor_field_get_sf_vec3f(size_field);
        boxes.push_back({parent_R,
                            parent_t,
                            Eigen::Vector3d(sv[0], sv[1], sv[2])});
    }

    return boxes;
}

/** Return the 8 corners of a single oriented box in vehicle frame. */
Corners8x3 robot_driver::RobotDriver::get_box_corners(const BoxInfo& box) {
    Eigen::Vector3d h = box.size / 2.0;
    double x = h.x(), y = h.y(), z = h.z();

    Corners8x3 corners_local;
    corners_local <<
            x, -y, -z,
            x,  y, -z,
            -x,  y, -z,
            -x, -y, -z,
            x, -y,  z,
            x,  y,  z,
            -x,  y,  z,
            -x, -y,  z;

    // Transform to vehicle frame
    return (box.R_local_to_vehicle * corners_local.transpose()).transpose().rowwise()
            + box.t_local_to_vehicle.transpose();
}

/** Compute an axis-aligned bounding box (in vehicle frame) over all sub-boxes. */
Corners8x3 robot_driver::RobotDriver::get_bounding_box(const std::vector<BoxInfo>& boxes)
{
    // Collect all corners
    Eigen::MatrixXd all(boxes.size() * 8, 3);
    for (size_t i = 0; i < boxes.size(); ++i)
        all.block<8, 3>(i * 8, 0) = get_box_corners(boxes[i]);

    Eigen::Vector3d mn = all.colwise().minCoeff();
    Eigen::Vector3d mx = all.colwise().maxCoeff();
    Eigen::Vector3d center = (mn + mx) / 2.0;
    Eigen::Vector3d h      = (mx - mn) / 2.0;
    double x = h.x(), y = h.y(), z = h.z();

    Corners8x3 corners;
    corners <<
            x, -y, -z,
            x,  y, -z,
            -x,  y, -z,
            -x, -y, -z,
            x, -y,  z,
            x,  y,  z,
            -x,  y,  z,
            -x, -y,  z;

    return corners.rowwise() + center.transpose();
}

/** Convert a list of 8-corner boxes (in lidar frame) to a MarkerArray. */
visualization_msgs::msg::MarkerArray robot_driver::RobotDriver::corners_to_marker_array(const std::vector<Corners8x3>& all_corners) {
    visualization_msgs::msg::MarkerArray ma;

    // Clear all previous markers
    visualization_msgs::msg::Marker clear_marker;
    clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear_marker);

    static const std::array<std::pair<int,int>, 12> edges = {{
        {0,1},{1,2},{2,3},{3,0},
        {4,5},{5,6},{6,7},{7,4},
        {0,4},{1,5},{2,6},{3,7}
    }};

    for (int obj_id = 0; obj_id < static_cast<int>(all_corners.size()); ++obj_id) {
        const Corners8x3& corners = all_corners[obj_id];

        visualization_msgs::msg::Marker m;
        m.header.frame_id = "lidar";
        m.header.stamp    = node->get_clock()->now();
        m.ns              = "bounding_boxes";
        m.id              = obj_id;
        m.type            = visualization_msgs::msg::Marker::LINE_LIST;
        m.action          = visualization_msgs::msg::Marker::ADD;
        m.scale.x         = 0.05;
        m.color.r         = 1.0f;
        m.color.g         = 0.0f;
        m.color.b         = 0.0f;
        m.color.a         = 1.0f;

        for (const auto& [s, e] : edges) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = corners(s, 0); p1.y = corners(s, 1); p1.z = corners(s, 2);
            p2.x = corners(e, 0); p2.y = corners(e, 1); p2.z = corners(e, 2);
            m.points.push_back(p1);
            m.points.push_back(p2);
        }

        ma.markers.push_back(m);
    }

    return ma;
}

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(robot_driver::RobotDriver, webots_ros2_driver::PluginInterface)