#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Vector3.h>

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

class WebotsTfPublisherNode : public rclcpp::Node {
public:
    WebotsTfPublisherNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("webots_tf_publisher", options)
    {
        // Declare parameters with default values
        this->declare_parameter<std::vector<double>>("translation", {0.0, 0.0, 0.0});
        this->declare_parameter<std::vector<double>>("rotation", {0.0, 0.0, 1.0, 0.0});
        this->declare_parameter<std::string>("frame_id", "map");
        this->declare_parameter<std::string>("child_frame_id", "lio_map");

        auto trans_vec = this->get_parameter("translation").as_double_array();
        auto rot_vec = this->get_parameter("rotation").as_double_array();
        frame_id_ = this->get_parameter("frame_id").as_string();
        child_frame_id_ = this->get_parameter("child_frame_id").as_string();

        double tx = (trans_vec.size() >= 3) ? trans_vec[0] : 0.0;
        double ty = (trans_vec.size() >= 3) ? trans_vec[1] : 0.0;
        double tz = (trans_vec.size() >= 3) ? trans_vec[2] : 0.0;

        double ax = (rot_vec.size() >= 4) ? rot_vec[0] : 0.0;
        double ay = (rot_vec.size() >= 4) ? rot_vec[1] : 0.0;
        double az = (rot_vec.size() >= 4) ? rot_vec[2] : 1.0;
        double angle_rad = (rot_vec.size() >= 4) ? rot_vec[3] : 0.0;

        // Convert Webots VRML Axis-Angle (ax, ay, az, angle) to Quaternion
        tf2::Vector3 axis(ax, ay, az);
        if (axis.length() < 1e-6) {
            axis = tf2::Vector3(0.0, 0.0, 1.0);
        } else {
            axis.normalize();
        }
        tf2::Quaternion q(axis, angle_rad);
        q.normalize();

        static_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->get_clock()->now();
        tf_msg.header.frame_id = frame_id_;
        tf_msg.child_frame_id = child_frame_id_;

        tf_msg.transform.translation.x = tx;
        tf_msg.transform.translation.y = ty;
        tf_msg.transform.translation.z = tz;

        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        static_broadcaster_->sendTransform(tf_msg);

        RCLCPP_INFO(this->get_logger(),
            "Published static TF [%s -> %s]:\n"
            "  Translation : [%.4f, %.4f, %.4f]\n"
            "  VRML Rotation: axis=[%.6f, %.6f, %.6f], angle=%.4f rad\n"
            "  Quaternion  : [x=%.6f, y=%.6f, z=%.6f, w=%.6f]",
            frame_id_.c_str(), child_frame_id_.c_str(),
            tx, ty, tz, ax, ay, az, angle_rad,
            q.x(), q.y(), q.z(), q.w());
    }

private:
    std::string frame_id_;
    std::string child_frame_id_;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_broadcaster_;
};

int main(int argc, char** argv) {
    std::vector<double> cli_trans;
    std::vector<double> cli_rot;
    std::string cli_frame;
    std::string cli_child;

    std::vector<std::string> ros_args;
    ros_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--trans" || arg == "-t") && i + 3 < argc) {
            cli_trans.push_back(std::stod(argv[++i]));
            cli_trans.push_back(std::stod(argv[++i]));
            cli_trans.push_back(std::stod(argv[++i]));
        } else if ((arg == "--rot" || arg == "-r") && i + 4 < argc) {
            cli_rot.push_back(std::stod(argv[++i]));
            cli_rot.push_back(std::stod(argv[++i]));
            cli_rot.push_back(std::stod(argv[++i]));
            cli_rot.push_back(std::stod(argv[++i]));
        } else if (arg == "--frame" && i + 1 < argc) {
            cli_frame = argv[++i];
        } else if (arg == "--child" && i + 1 < argc) {
            cli_child = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ros2 run autonomous_drive webots_tf_publisher [options]\n\n"
                      << "Options:\n"
                      << "  --trans, -t <x> <y> <z>        Webots translation\n"
                      << "  --rot, -r <ax> <ay> <az> <rad> Webots VRML Axis-Angle rotation\n"
                      << "  --frame <frame_id>             Parent frame ID (default: map)\n"
                      << "  --child <child_frame_id>       Child frame ID (default: lio_map)\n"
                      << "  --ros-args                     ROS 2 standard arguments\n\n"
                      << "Example:\n"
                      << "  ros2 run autonomous_drive webots_tf_publisher \\\n"
                      << "    --trans 56.8755 -65.2634 0.400134 \\\n"
                      << "    --rot -0.0013009 3.97107e-06 0.999999 -1.0944 \\\n"
                      << "    --frame map --child lio_map\n";
            return 0;
        } else {
            ros_args.push_back(arg);
        }
    }

    std::vector<char*> c_argv;
    for (auto& s : ros_args) {
        c_argv.push_back(const_cast<char*>(s.c_str()));
    }
    int c_argc = static_cast<int>(c_argv.size());
    rclcpp::init(c_argc, c_argv.data());

    rclcpp::NodeOptions options;
    std::vector<rclcpp::Parameter> params;
    if (!cli_trans.empty()) {
        params.emplace_back("translation", cli_trans);
    }
    if (!cli_rot.empty()) {
        params.emplace_back("rotation", cli_rot);
    }
    if (!cli_frame.empty()) {
        params.emplace_back("frame_id", cli_frame);
    }
    if (!cli_child.empty()) {
        params.emplace_back("child_frame_id", cli_child);
    }
    if (!params.empty()) {
        options.parameter_overrides(params);
    }

    auto node = std::make_shared<WebotsTfPublisherNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
