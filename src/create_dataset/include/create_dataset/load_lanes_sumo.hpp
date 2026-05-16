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

    while (std::getline(file, line)) {
        if (remaining_pts > 0) {
            // Parse a point line: "x y"
            std::istringstream ss(line);
            Point pt;
            ss >> pt.x >> pt.y;

            if (current_side == "left_line")
                boundaries[current_lane_id].left_line.push_back(pt);
            else
                boundaries[current_lane_id].right_line.push_back(pt);

            --remaining_pts;
        } else {
            // Parse a header line: "lane_id|side|n_points"
            std::istringstream ss(line);
            std::string token;

            std::getline(ss, current_lane_id, '|');
            std::getline(ss, current_side,    '|');
            std::getline(ss, token,           '|');
            remaining_pts = std::stoi(token);

            // Pre-allocate
            auto& bnd = boundaries[current_lane_id];
            if (current_side == "left_line")
                bnd.left_line.reserve(remaining_pts);
            else
                bnd.right_line.reserve(remaining_pts);
        }
    }

    return boundaries;
}