#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <ackermann_msgs/msg/ackermann_drive.hpp>
#include <opencv2/opencv.hpp>

using std::placeholders::_1;

class LaneFollower : public rclcpp::Node
{
public:
    LaneFollower() : Node("lane_follower")
    {
        ackermann_pub_ =
            this->create_publisher<ackermann_msgs::msg::AckermannDrive>(
                "cmd_ackermann", 1);

        auto qos = rclcpp::SensorDataQoS();
        qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);

        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "vehicle/camera/image_color",
            qos,
            std::bind(&LaneFollower::imageCallback, this, _1));

        RCLCPP_INFO(this->get_logger(), "LaneFollower node started.");
    }

private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // Convert ROS Image -> OpenCV Mat (RGBA8 assumed)
        cv::Mat img(msg->height, msg->width, CV_8UC4,
                    const_cast<unsigned char*>(msg->data.data()));

        // Crop region of interest
        cv::Rect roi(0, 160, msg->width, 30);
        cv::Mat cropped = img(roi);

        // Convert RGBA -> RGB -> HSV
        cv::Mat rgb, hsv;
        cv::cvtColor(cropped, rgb, cv::COLOR_RGBA2RGB);
        cv::cvtColor(rgb, hsv, cv::COLOR_RGB2HSV);

        // Threshold lane color
        cv::Scalar lower(50, 110, 150);
        cv::Scalar upper(120, 255, 255);
        cv::Mat mask;
        cv::inRange(hsv, lower, upper, mask);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        ackermann_msgs::msg::AckermannDrive cmd;
        cmd.speed = 20.0;
        cmd.steering_angle = 0.0;

        if (!contours.empty())
        {
            auto largest = std::max_element(
                contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a,
                   const std::vector<cv::Point>& b)
                {
                    return cv::contourArea(a) < cv::contourArea(b);
                });

            cv::Moments m = cv::moments(*largest);

            if (m.m00 != 0)
            {
                int center_x = static_cast<int>(m.m10 / m.m00);

                double error = center_x - 190;
                const double CONTROL_COEFFICIENT = 0.0005;

                cmd.steering_angle = error * CONTROL_COEFFICIENT;
            }
        }

        ackermann_pub_->publish(cmd);
    }

    rclcpp::Publisher<ackermann_msgs::msg::AckermannDrive>::SharedPtr ackermann_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LaneFollower>());
    rclcpp::shutdown();
    return 0;
}