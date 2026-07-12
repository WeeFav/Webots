#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <visualization_msgs/msg/marker.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

class PurePursuitController : public rclcpp::Node {
public:
  PurePursuitController() : Node("pure_pursuit_controller") {
    // Declare parameters
    this->declare_parameter<double>("lookahead_distance", 4.0); // meters
    this->declare_parameter<double>("wheelbase", 2.94);          // meters

    // Get parameters
    lookahead_distance_ = this->get_parameter("lookahead_distance").as_double();
    wheelbase_ = this->get_parameter("wheelbase").as_double();

    // Set up parameter change callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
          if (param.get_name() == "lookahead_distance") {
            lookahead_distance_ = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated lookahead_distance: %.2f m", lookahead_distance_);
          } else if (param.get_name() == "wheelbase") {
            wheelbase_ = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated wheelbase: %.2f m", wheelbase_);
          }
        }
        return result;
      });

    // Publishers and Subscriptions
    path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
      "/centerline_waypoints", 10,
      std::bind(&PurePursuitController::pathCallback, this, std::placeholders::_1));

    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/vehicle/world_pose", 10,
      std::bind(&PurePursuitController::poseCallback, this, std::placeholders::_1));

    steering_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vehicle/steering_angle", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>("/vehicle/lookahead_marker", 10);

    RCLCPP_INFO(this->get_logger(), "Pure Pursuit Node Initialized. Lookahead: %.2f m | Wheelbase: %.2f m",
                lookahead_distance_, wheelbase_);
  }

private:
  void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
    if (msg->poses.size() < 2) {
      path_ = msg;
      return;
    }

    auto interpolated_path = std::make_shared<nav_msgs::msg::Path>();
    interpolated_path->header = msg->header;

    const double step_size = 0.1; // 10 cm resolution

    for (size_t i = 0; i + 1 < msg->poses.size(); ++i) {
      const auto & p1 = msg->poses[i];
      const auto & p2 = msg->poses[i+1];

      double dx = p2.pose.position.x - p1.pose.position.x;
      double dy = p2.pose.position.y - p1.pose.position.y;
      double dz = p2.pose.position.z - p1.pose.position.z;
      double len = std::sqrt(dx*dx + dy*dy + dz*dz);

      interpolated_path->poses.push_back(p1);

      if (len > step_size) {
        int num_steps = static_cast<int>(len / step_size);
        for (int j = 1; j < num_steps; ++j) {
          double t = static_cast<double>(j) / num_steps;
          geometry_msgs::msg::PoseStamped inter_pose;
          inter_pose.header = msg->header;
          inter_pose.pose.position.x = p1.pose.position.x + t * dx;
          inter_pose.pose.position.y = p1.pose.position.y + t * dy;
          inter_pose.pose.position.z = p1.pose.position.z + t * dz;

          // Slerp for orientation
          Eigen::Quaterniond q1(p1.pose.orientation.w, p1.pose.orientation.x, p1.pose.orientation.y, p1.pose.orientation.z);
          Eigen::Quaterniond q2(p2.pose.orientation.w, p2.pose.orientation.x, p2.pose.orientation.y, p2.pose.orientation.z);
          Eigen::Quaterniond q_inter = q1.slerp(t, q2);

          inter_pose.pose.orientation.x = q_inter.x();
          inter_pose.pose.orientation.y = q_inter.y();
          inter_pose.pose.orientation.z = q_inter.z();
          inter_pose.pose.orientation.w = q_inter.w();

          interpolated_path->poses.push_back(inter_pose);
        }
      }
    }
    interpolated_path->poses.push_back(msg->poses.back());

    path_ = interpolated_path;
    RCLCPP_INFO(this->get_logger(), "Received new path with %zu poses, interpolated to %zu poses.",
                msg->poses.size(), path_->poses.size());
  }

  void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    if (!path_ || path_->poses.empty()) {
      // No path to follow yet
      return;
    }

    double vehicle_x = msg->pose.position.x;
    double vehicle_y = msg->pose.position.y;

    // Convert quaternion to yaw angle
    double qx = msg->pose.orientation.x;
    double qy = msg->pose.orientation.y;
    double qz = msg->pose.orientation.z;
    double qw = msg->pose.orientation.w;
    double vehicle_yaw = std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));

    // 1. Find lookahead point by searching forward from the closest point
    size_t closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < path_->poses.size(); ++i) {
      double dx = path_->poses[i].pose.position.x - vehicle_x;
      double dy = path_->poses[i].pose.position.y - vehicle_y;
      double dist = std::sqrt(dx*dx + dy*dy);
      if (dist < min_dist) {
        min_dist = dist;
        closest_idx = i;
      }
    }

    geometry_msgs::msg::Point target_pt;
    bool found_target = false;
    for (size_t i = closest_idx; i < path_->poses.size(); ++i) {
      double dx = path_->poses[i].pose.position.x - vehicle_x;
      double dy = path_->poses[i].pose.position.y - vehicle_y;
      double dist = std::sqrt(dx*dx + dy*dy);

      // We look for the first point whose distance is >= lookahead_distance
      if (dist >= lookahead_distance_) {
        target_pt = path_->poses[i].pose.position;
        found_target = true;
        break;
      }
    }

    // Fallback: If no point is far enough, use the last point of the path
    if (!found_target) {
      target_pt = path_->poses.back().pose.position;
    }

    // Publish lookahead point marker
    visualization_msgs::msg::Marker marker;
    marker.header.stamp = this->now();
    marker.header.frame_id = "map";
    marker.ns = "lookahead";
    marker.id = 0;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position = target_pt;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.6;
    marker.scale.y = 0.6;
    marker.scale.z = 0.6;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    marker_pub_->publish(marker);

    // 2. Transform the lookahead point to the vehicle's local coordinate frame
    double dx = target_pt.x - vehicle_x;
    double dy = target_pt.y - vehicle_y;

    // x is forward, y is left
    double x_local = dx * std::cos(vehicle_yaw) + dy * std::sin(vehicle_yaw);
    double y_local = -dx * std::sin(vehicle_yaw) + dy * std::cos(vehicle_yaw);

    // 3. Compute curvature and steering angle
    double dist = std::sqrt(dx*dx + dy*dy);
    if (dist < 0.1) dist = 0.1; // avoid division by zero

    double curvature = 2.0 * y_local / (dist * dist);
    double steering_angle = std::atan2(wheelbase_ * curvature, 1.0);

    // Clamp steering angle to vehicle limits [-0.5, 0.5] rad
    steering_angle = std::max(-0.5, std::min(0.5, steering_angle));

    // 4. Publish steering angle
    std_msgs::msg::Float64 steer_msg;
    steer_msg.data = -steering_angle;
    steering_pub_->publish(steer_msg);

    // RCLCPP_INFO(this->get_logger(),
    //              "Pose: (%.2f, %.2f) Yaw: %.2f | Lookahead point: (%.2f, %.2f) | Local: (%.2f, %.2f) | Steer: %.3f rad",
    //              vehicle_x, vehicle_y, vehicle_yaw, target_pt.x, target_pt.y, x_local, y_local, steering_angle);
  }

  // Parameters
  double lookahead_distance_;
  double wheelbase_;

  // Cache
  nav_msgs::msg::Path::SharedPtr path_;

  // ROS 2 interfaces
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PurePursuitController>());
  rclcpp::shutdown();
  return 0;
}
