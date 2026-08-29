#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/impl/voxel_grid.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include <string>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

class PcdPublisherNode : public rclcpp::Node {
public:
    PcdPublisherNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("pcd_publisher", options)
    {
        // Declare ROS 2 parameters with default values
        this->declare_parameter<std::string>("pcd_file", "");
        this->declare_parameter<std::string>("topic", "/pcd_map");
        this->declare_parameter<std::string>("frame_id", "map");
        this->declare_parameter<double>("publish_rate", 0.2); // Default: 0.2 Hz (re-publish every 5 seconds)
        this->declare_parameter<double>("leaf_size", 0.0);    // Default: 0.0 (no downsampling filter)
        this->declare_parameter<bool>("latch", true);

        // Retrieve parameters
        pcd_file_ = this->get_parameter("pcd_file").as_string();
        topic_name_ = this->get_parameter("topic").as_string();
        frame_id_ = this->get_parameter("frame_id").as_string();
        publish_rate_ = this->get_parameter("publish_rate").as_double();
        leaf_size_ = this->get_parameter("leaf_size").as_double();
        latch_ = this->get_parameter("latch").as_bool();

        if (pcd_file_.empty()) {
            RCLCPP_ERROR(this->get_logger(), "No PCD file specified! Set parameter 'pcd_file' or pass --pcd <file.pcd>");
            return;
        }

        // Configure QoS profile (transient local / latched for static map publishing)
        rclcpp::QoS qos(1);
        if (latch_) {
            qos.transient_local();
            qos.reliable();
        }

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(topic_name_, qos);

        // Load and process PCD file
        if (!loadPCDFile()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", pcd_file_.c_str());
            return;
        }

        // Publish initially
        publishPointCloud();

        // Setup timer only if publish_rate > 0.0
        if (publish_rate_ > 0.0) {
            RCLCPP_INFO(this->get_logger(), "Publishing PointCloud2 to topic '%s' with frame_id '%s' at %.1f Hz",
                        topic_name_.c_str(), frame_id_.c_str(), publish_rate_);
            auto interval = std::chrono::duration<double>(1.0 / publish_rate_);
            timer_ = this->create_wall_timer(
                std::chrono::duration_cast<std::chrono::milliseconds>(interval),
                std::bind(&PcdPublisherNode::publishPointCloud, this)
            );
        } else {
            RCLCPP_INFO(this->get_logger(), "Published static map to topic '%s' (transient local / latched). Ready for RViz2!",
                        topic_name_.c_str());
        }
    }

private:
    bool loadPCDFile() {
        pcl::PointCloud<pcl::PointXYZI>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_, *raw_cloud) < 0) {
            return false;
        }

        size_t original_count = raw_cloud->points.size();

        // 1. Remove NaNs and invalid points
        std::vector<int> valid_indices;
        pcl::removeNaNFromPointCloud(*raw_cloud, *raw_cloud, valid_indices);

        // 2. Filter spatial outliers (e.g. corrupted points beyond 3000m from origin)
        pcl::PointCloud<pcl::PointXYZI>::Ptr clean_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        clean_cloud->reserve(raw_cloud->points.size());
        for (const auto& pt : raw_cloud->points) {
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z) &&
                std::abs(pt.x) < 3000.0f && std::abs(pt.y) < 3000.0f && std::abs(pt.z) < 1000.0f) {
                clean_cloud->points.push_back(pt);
            }
        }
        clean_cloud->width = clean_cloud->points.size();
        clean_cloud->height = 1;
        clean_cloud->is_dense = true;

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*clean_cloud, min_pt, max_pt);
        RCLCPP_INFO(this->get_logger(), "Map Bounding Box: X[%.1f, %.1f] Y[%.1f, %.1f] Z[%.1f, %.1f]",
                    min_pt[0], max_pt[0], min_pt[1], max_pt[1], min_pt[2], max_pt[2]);

        pcl::PointCloud<pcl::PointXYZI>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZI>());

        // 3. Apply Chunked VoxelGrid downsampling if leaf_size > 0
        if (leaf_size_ > 0.0) {
            // To prevent 32-bit index integer overflow in pcl::VoxelGrid when processing large maps:
            // Process the cloud in spatial grid chunks (e.g. 500m x 500m blocks)
            float chunk_size = 500.0f;

            // Bucket points into chunks
            std::map<std::pair<int, int>, pcl::PointCloud<pcl::PointXYZI>::Ptr> chunks;
            for (const auto& pt : clean_cloud->points) {
                int cx = static_cast<int>(std::floor(pt.x / chunk_size));
                int cy = static_cast<int>(std::floor(pt.y / chunk_size));
                auto key = std::make_pair(cx, cy);
                if (chunks.find(key) == chunks.end()) {
                    chunks[key] = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
                }
                chunks[key]->points.push_back(pt);
            }

            pcl::VoxelGrid<pcl::PointXYZI> vg;
            vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), static_cast<float>(leaf_size_));

            for (auto& [key, chunk_cloud] : chunks) {
                if (chunk_cloud->empty()) continue;
                chunk_cloud->width = chunk_cloud->points.size();
                chunk_cloud->height = 1;
                chunk_cloud->is_dense = true;

                pcl::PointCloud<pcl::PointXYZI> filtered_chunk;
                vg.setInputCloud(chunk_cloud);
                vg.filter(filtered_chunk);
                *final_cloud += filtered_chunk;
            }

            RCLCPP_INFO(this->get_logger(), "Loaded PCD: %s | Downsampled from %zu to %zu points (leaf_size = %.2f m)",
                        pcd_file_.c_str(), original_count, final_cloud->points.size(), leaf_size_);
        } else {
            final_cloud = clean_cloud;
            RCLCPP_INFO(this->get_logger(), "Loaded PCD: %s (%zu points)", pcd_file_.c_str(), final_cloud->points.size());
        }

        pcl::toROSMsg(*final_cloud, cloud_msg_);
        cloud_msg_.header.frame_id = frame_id_;
        return true;
    }

    void publishPointCloud() {
        if (!publisher_) return;
        cloud_msg_.header.stamp = this->now();
        publisher_->publish(cloud_msg_);
    }

    std::string pcd_file_;
    std::string topic_name_;
    std::string frame_id_;
    double publish_rate_{0.0};
    double leaf_size_{0.0};
    bool latch_{true};

    sensor_msgs::msg::PointCloud2 cloud_msg_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    std::string cli_pcd;
    std::string cli_topic;
    std::string cli_frame;
    double cli_rate = -1.0;
    double cli_leaf = -1.0;

    std::vector<std::string> ros_args;
    ros_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--pcd" || arg == "-f") && i + 1 < argc) {
            cli_pcd = argv[++i];
        } else if (arg == "--topic" && i + 1 < argc) {
            cli_topic = argv[++i];
        } else if (arg == "--frame" && i + 1 < argc) {
            cli_frame = argv[++i];
        } else if (arg == "--rate" && i + 1 < argc) {
            cli_rate = std::stod(argv[++i]);
        } else if ((arg == "--leaf-size" || arg == "--leaf") && i + 1 < argc) {
            cli_leaf = std::stod(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: ros2 run autonomous_drive pcd_publisher [options]\n\n"
                      << "Options:\n"
                      << "  --pcd, -f <file.pcd>       Path to PCD file\n"
                      << "  --topic <topic_name>       Topic to publish (default: /pcd_map)\n"
                      << "  --frame <frame_id>         Frame ID for header (default: map)\n"
                      << "  --leaf-size, --leaf <m>    Voxel grid downsample leaf size in meters (e.g. 0.2, 0.4)\n"
                      << "  --rate <hz>                Publish rate in Hz (default: 0.2 = publish every 5 seconds)\n"
                      << "  --ros-args                 ROS 2 standard arguments\n\n"
                      << "Examples:\n"
                      << "  ros2 run autonomous_drive pcd_publisher --pcd merged_map.pcd --leaf 0.2\n";
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
    if (!cli_pcd.empty()) {
        params.emplace_back("pcd_file", cli_pcd);
    }
    if (!cli_topic.empty()) {
        params.emplace_back("topic", cli_topic);
    }
    if (!cli_frame.empty()) {
        params.emplace_back("frame_id", cli_frame);
    }
    if (cli_rate >= 0.0) {
        params.emplace_back("publish_rate", cli_rate);
    }
    if (cli_leaf > 0.0) {
        params.emplace_back("leaf_size", cli_leaf);
    }
    if (!params.empty()) {
        options.parameter_overrides(params);
    }

    auto node = std::make_shared<PcdPublisherNode>(options);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
