#include <torch/torch.h>
#include <torch/script.h>
 
#include <opencv2/opencv.hpp>
 
#include <vector>
#include <string>
#include <stdexcept>

#include "lane_detection.hpp"

static const std::vector<int> CARLA_ROW_ANCHOR = {
    64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
    116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
    168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
    220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
    272, 276, 280, 284
};

void lane_detection::LaneDetection::LaneDetection() : Node("lane_detection") {
    this->declare_parameter<std::string>("model_path",    "model.pt");
    this->declare_parameter<int>        ("griding_num",   100);
    this->declare_parameter<int>        ("num_lanes",     4);
    this->declare_parameter<int>        ("img_w",         800);
    this->declare_parameter<int>        ("img_h",         450);
    this->declare_parameter<std::string>("image_topic",   "/camera/image_raw");
    this->declare_parameter<std::string>("output_topic",  "/lane_detection/image");

    model_path   = this->get_parameter("model_path").as_string();
    griding_num  = this->get_parameter("griding_num").as_int();
    num_lanes    = this->get_parameter("num_lanes").as_int();
    img_w        = this->get_parameter("img_w").as_int();
    img_h        = this->get_parameter("img_h").as_int();

    cls_num_per_lane = static_cast<int>(CARLA_ROW_ANCHOR.size()); 

    col_sample.resize(griding_num);
    for (int i = 0; i < griding_num; ++i) {
        col_sample[i] = static_cast<float>(i) * (800.0f - 1.0f) / static_cast<float>(griding_num - 1);
    }
    col_sample_w = col_sample[1] - col_sample[0];

    // Load TorchScript model
    RCLCPP_INFO(this->get_logger(), "Loading model from: %s", model_path.c_str());
    try {
        model = torch::jit::load(model_path);
        model.eval();
        if (torch::cuda::is_available()) {
            model.to(torch::kCUDA);
            use_cuda = true;
            RCLCPP_INFO(this->get_logger(), "Running inference on CUDA.");
        } else {
            RCLCPP_WARN(this->get_logger(), "CUDA not available; running on CPU.");
        }
    } catch (const c10::Error& e) {
        RCLCPP_FATAL(this->get_logger(), "Failed to load model: %s", e.what());
        throw;
    }  

    // ImageNet normalisation constants
    norm_mean = {0.485f, 0.456f, 0.406f};
    norm_std  = {0.229f, 0.224f, 0.225f}; 

    // ROS2 pub / sub
    std::string image_topic  = this->get_parameter("image_topic").as_string();
    std::string output_topic = this->get_parameter("output_topic").as_string();

    image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        image_topic, 10,
        std::bind(&LaneDetectionNode::imageCallback, this, std::placeholders::_1));

    image_pub = this->create_publisher<sensor_msgs::msg::Image>(output_topic, 10);

    RCLCPP_INFO(this->get_logger(), "Lane detection node ready.\n  Subscribing : %s\n  Publishing  : %s", in_topic.c_str(), out_topic.c_str());   
}

void lane_detection::LaneDetection::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
    // -- 1. Decode ROS image -----------------------------------------
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat original = cv_ptr->image.clone(); // keep for annotation

    // -- 2. Pre-process: resize → float → CHW → normalise ------------
    //    Mirrors transforms.Compose([Resize(288,800), ToTensor(), Normalize(...)])
    cv::Mat resized;
    cv::resize(original, resized, cv::Size(800, 288));

    // BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // HWC uint8 → CHW float32, scaled to [0,1]
    torch::Tensor input_tensor = torch::from_blob(
        rgb.data,
        {1, 288, 800, 3},
        torch::kByte
    ).to(torch::kFloat32).div(255.0f);

    // NHWC → NCHW
    input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

    // Normalise per-channel: (x - mean) / std
    for (int c = 0; c < 3; ++c) {
        input_tensor[0][c] = (input_tensor[0][c] - norm_mean[c]) / norm_std[c];
    }

    if (use_cuda) {
        input_tensor = input_tensor.to(torch::kCUDA);
    }

    // -- 3. Forward pass ---------------------------------------------
    torch::Tensor detection;
    try {
        torch::NoGradGuard no_grad;
        // model returns a dict-like IValue; 'det' key mirrors out['det']
        auto output = model_.forward({input_tensor}).toGenericDict();
        detection   = output.at("det").toTensor().to(torch::kCPU);
    } catch (const c10::Error& e) {
        RCLCPP_ERROR(this->get_logger(), "Inference error: %s", e.what());
        return;
    }

    // -- 4. Post-process (mirrors demo.py exactly) -------------------
    // detection shape: (1, griding_num+1, cls_num_per_lane, num_lanes)
    // Take batch index 0 → (griding_num+1, cls_num_per_lane, num_lanes)
    auto out_j = detection[0]; // shape: [G+1, R, L]

    // Flip rows: out_j = out_j[:, ::-1, :]
    out_j = out_j.flip(1);

    // Softmax over griding_num classes (drop the background class at index G)
    // prob shape: [G, R, L]
    auto prob = torch::softmax(out_j.slice(0, 0, griding_num), 0);

    // Weighted expectation: idx = [1..griding_num] reshaped to [G,1,1]
    auto idx = torch::arange(1, griding_num + 1,
                                torch::TensorOptions().dtype(torch::kFloat32))
                    .reshape({griding_num, 1, 1});
    // loc shape: [R, L]
    auto loc = (prob * idx).sum(0);

    // argmax over all G+1 classes (including background)
    auto argmax = out_j.argmax(0); // [R, L]

    // Where argmax == griding_num (background class), set loc to 0
    loc = torch::where(argmax == griding_num, torch::zeros_like(loc), loc);

    // Copy to CPU accessor
    auto loc_acc    = loc.accessor<float, 2>();    // [R, L]

    // -- 5. Draw lane points on the original image -------------------
    //    Mirrors the cv2.circle loop in demo.py
    cv::Mat vis = original.clone();
    int R = cls_num_per_lane;   // num row anchors
    int L = num_lanes;

    for (int lane = 0; lane < L; ++lane) {
        // Count non-zero predictions for this lane
        int valid_count = 0;
        for (int row = 0; row < R; ++row) {
            if (loc_acc[row][lane] != 0.0f) ++valid_count;
        }
        if (valid_count <= 2) continue; // skip sparse lanes

        for (int row = 0; row < R; ++row) {
            float x_norm = loc_acc[row][lane];
            if (x_norm <= 0.0f) continue;

            // Map from grid index to pixel coordinates
            // x: int(x_norm * col_sample_w * img_w / 800) - 1
            int px = static_cast<int>(x_norm * col_sample_w *
                                        static_cast<float>(img_w) / 800.0f) - 1;

            // y: int(img_h * (row_anchor[R-1-row] / 288)) - 1
            float anchor_ratio = static_cast<float>(CARLA_ROW_ANCHOR[R - 1 - row]) / 288.0f;
            int py = static_cast<int>(static_cast<float>(img_h) * anchor_ratio) - 1;

            // Clamp to image bounds
            px = std::clamp(px, 0, img_w - 1);
            py = std::clamp(py, 0, img_h - 1);

            cv::circle(vis, cv::Point(px, py), 5, cv::Scalar(0, 255, 0), -1);
        }
    }

    // -- 6. Publish annotated image ----------------------------------
    auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", vis).toImageMsg();
    image_pub->publish(*out_msg);
}