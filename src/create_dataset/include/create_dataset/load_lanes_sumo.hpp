#pragma once

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "robot_driver.hpp"

inline std::unordered_map<std::string, LaneBoundary> load_lane_boundaries(const std::string& filepath) {
    std::unordered_map<std::string, LaneBoundary> boundaries;

    std::ifstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Could not open file: " + filepath);

    std::string line;
    std::string current_lane_id;
    std::string current_side;
    int remaining_pts = 0;
    int total_pts     = 0;

    while (std::getline(file, line)) {
        if (remaining_pts > 0) {
            // Parse a point line: "x y"
            std::istringstream ss(line);
            double x, y;
            ss >> x >> y;

            int row = total_pts - remaining_pts;  // current row index

            if (current_side == "left_line") {
                boundaries[current_lane_id].left_line(row, 0) = x;
                boundaries[current_lane_id].left_line(row, 1) = y;
                boundaries[current_lane_id].left_line(row, 2) = 0.02;
            } else {
                boundaries[current_lane_id].right_line(row, 0) = x;
                boundaries[current_lane_id].right_line(row, 1) = y;
                boundaries[current_lane_id].right_line(row, 2) = 0.02;
            }

            --remaining_pts;
        } else {
            // Parse a header line: "lane_id|side|n_points"
            std::istringstream ss(line);
            std::string token;

            std::getline(ss, current_lane_id, '|');
            std::getline(ss, current_side,    '|');
            std::getline(ss, token,           '|');
            total_pts    = std::stoi(token);
            remaining_pts = total_pts;

            // Pre-allocate the Eigen matrix (N, 3)
            auto& bnd = boundaries[current_lane_id];
            if (current_side == "left_line")
                bnd.left_line  = Eigen::MatrixXd(total_pts, 3);
            else
                bnd.right_line = Eigen::MatrixXd(total_pts, 3);
        }
    }
    return boundaries;
}