#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
 
#include <torch/torch.h>
#include <torch/script.h>
 
#include <opencv2/opencv.hpp>
 
#include <vector>
#include <string>
#include <stdexcept>

namespace lane_detection {
class LaneDetection : public rclcpp::Node {
public:
    LaneDetectionNode();
private:
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    torch::jit::script::Module model;
    bool use_cuda_ = false;
 
    std::string model_path;
    int griding_num;
    int num_lanes;
    int cls_num_per_lane;
    int img_w_, img_h;
 
    std::vector<float> col_sample;
    float col_sample_w;
 
    std::vector<float> norm_mean;
    std::vector<float> norm_std;
 
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr     image_pub;  
}
}