#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <chrono>
#include <algorithm>
#include <memory>

class PIDController : public rclcpp::Node {
public:
  PIDController() : Node("pid_controller") {
    // Declare parameters
    this->declare_parameter<double>("target_speed", 5.0); // m/s (approx 18 km/h)
    this->declare_parameter<double>("Kp", 0.8);
    this->declare_parameter<double>("Ki", 0.05);
    this->declare_parameter<double>("Kd", 0.1);

    // Get initial parameter values
    target_speed_ = this->get_parameter("target_speed").as_double();
    Kp_ = this->get_parameter("Kp").as_double();
    Ki_ = this->get_parameter("Ki").as_double();
    Kd_ = this->get_parameter("Kd").as_double();

    // Set up parameter change callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
          if (param.get_name() == "target_speed") {
            target_speed_ = param.as_double();
            RCLCPP_INFO(this->get_logger(), "Updated target_speed: %.2f m/s", target_speed_);
          } else if (param.get_name() == "Kp") {
            Kp_ = param.as_double();
          } else if (param.get_name() == "Ki") {
            Ki_ = param.as_double();
          } else if (param.get_name() == "Kd") {
            Kd_ = param.as_double();
          }
        }
        return result;
      });

    // Publishers and Subscriptions
    velocity_sub_ = this->create_subscription<std_msgs::msg::Float64>(
      "/vehicle/current_velocity", 10,
      std::bind(&PIDController::velocityCallback, this, std::placeholders::_1));

    throttle_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vehicle/throttle", 10);
    brake_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vehicle/brake", 10);

    last_time_ = this->now();
    RCLCPP_INFO(this->get_logger(), "PID Controller Node Initialized. Target Speed: %.2f m/s", target_speed_);
  }

private:
  void velocityCallback(const std_msgs::msg::Float64::SharedPtr msg) {
    double current_speed = msg->data;
    rclcpp::Time now = this->now();
    double dt = (now - last_time_).seconds();
    
    // Fallback if dt is negative or close to 0 (e.g. startup or time reset)
    if (dt <= 0.0 || dt > 1.0) {
      dt = 0.032; 
    }
    last_time_ = now;

    double error = target_speed_ - current_speed;
    
    // Accumulate integral with anti-windup clamping
    integral_ += error * dt;
    integral_ = std::max(-1.0, std::min(1.0, integral_));

    double derivative = (error - prev_error_) / dt;
    prev_error_ = error;

    double output = Kp_ * error + Ki_ * integral_ + Kd_ * derivative;

    std_msgs::msg::Float64 throttle_msg;
    std_msgs::msg::Float64 brake_msg;

    if (output >= 0.0) {
      throttle_msg.data = std::min(1.0, output);
      brake_msg.data = 0.0;
    } else {
      throttle_msg.data = 0.0;
      // Proportional braking (negate output and clamp)
      brake_msg.data = std::min(1.0, -output);
    }

    throttle_pub_->publish(throttle_msg);
    brake_pub_->publish(brake_msg);

    RCLCPP_DEBUG(this->get_logger(), 
                 "Target: %.2f | Curr: %.2f | Err: %.2f | Out: %.2f | Throttle: %.2f | Brake: %.2f",
                 target_speed_, current_speed, error, output, throttle_msg.data, brake_msg.data);
  }

  // Parameters
  double target_speed_;
  double Kp_;
  double Ki_;
  double Kd_;

  // PID State variables
  double integral_{0.0};
  double prev_error_{0.0};
  rclcpp::Time last_time_;

  // ROS 2 interfaces
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr velocity_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr throttle_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr brake_pub_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PIDController>());
  rclcpp::shutdown();
  return 0;
}
