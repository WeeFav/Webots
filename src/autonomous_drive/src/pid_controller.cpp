#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <chrono>
#include <algorithm>
#include <memory>

class PIDController : public rclcpp::Node {
public:
  PIDController() : Node("pid_controller") {
    // Declare parameters
    this->declare_parameter<double>("Kp", 0.4);
    this->declare_parameter<double>("Ki", 0.05);
    this->declare_parameter<double>("Kd", 0.1);

    // Get initial parameter values
    Kp_ = this->get_parameter("Kp").as_double();
    Ki_ = this->get_parameter("Ki").as_double();
    Kd_ = this->get_parameter("Kd").as_double();

    // Set up parameter change callback
    param_callback_handle_ = this->add_on_set_parameters_callback(
      [this](const std::vector<rclcpp::Parameter> &parameters) {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : parameters) {
          if (param.get_name() == "Kp") {
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

    target_speed_sub_ = this->create_subscription<std_msgs::msg::Float64>(
      "/vehicle/target_speed", 10,
      [this](const std_msgs::msg::Float64::SharedPtr msg) {
        target_speed_ = msg->data;
      });

    throttle_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vehicle/throttle", 10);
    brake_pub_ = this->create_publisher<std_msgs::msg::Float64>("/vehicle/brake", 10);

    last_time_ = this->now();
    RCLCPP_INFO(this->get_logger(), "PID Controller Node Initialized.");
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

    // Error deadbands (m/s)
    const double accel_deadband = 0.1;   // Don't accelerate if within 0.2 m/s
    const double brake_deadband = -1.0;  // Don't brake unless 1 m/s over target

    // PID
    integral_ += error * dt;
    integral_ = std::clamp(integral_, -1.0, 1.0);

    double derivative = (error - prev_error_) / dt;
    prev_error_ = error;

    double output = Kp_ * error + Ki_ * integral_ + Kd_ * derivative;

    std_msgs::msg::Float64 throttle_msg;
    std_msgs::msg::Float64 brake_msg;

    if (error > accel_deadband) {
        // Below target -> accelerate
        throttle_msg.data = std::clamp(output, 0.0, 1.0);
        brake_msg.data = 0.0;
    }
    else if (error < brake_deadband) {
        // Well above target -> brake
        throttle_msg.data = 0.0;
        brake_msg.data = std::clamp(-output, 0.0, 1.0);
    }
    else {
        // Close enough -> coast
        throttle_msg.data = 0.0;
        brake_msg.data = 0.0;
    }

    throttle_pub_->publish(throttle_msg);
    brake_pub_->publish(brake_msg);

    RCLCPP_DEBUG(this->get_logger(), 
                 "Target: %.2f | Curr: %.2f | Err: %.2f | Out: %.2f | Throttle: %.2f | Brake: %.2f",
                 target_speed_, current_speed, error, output, throttle_msg.data, brake_msg.data);
  }

  // Parameters
  double target_speed_{5.0};
  double Kp_;
  double Ki_;
  double Kd_;

  // PID State variables
  double integral_{0.0};
  double prev_error_{0.0};
  rclcpp::Time last_time_;

  // ROS 2 interfaces
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr velocity_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr target_speed_sub_;
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
