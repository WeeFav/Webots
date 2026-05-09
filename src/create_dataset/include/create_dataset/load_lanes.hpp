#pragma once

// load_lanes.hpp
// Drop this header into your project and call load_lanes_txt() wherever you
// previously called the Python extract_lanes() function.
//
// Matches the format written by export_lanes.py:
//
//   <num_lanes>
//   <num_points_lane_0>
//   x0 y0 z0
//   x1 y1 z1
//   ...
//   <num_points_lane_1>
//   ...

#include <Eigen/Core>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Load pre-computed lane centre-lines from a text file.
 *
 * @param  path   Path to the lanes.txt file produced by export_lanes.py.
 * @return        Vector of lanes; each lane is an (N × 3) Eigen matrix whose
 *                rows are (x, y, z) world-frame points.
 *
 * @throws std::runtime_error  if the file cannot be opened or is malformed.
 */
inline std::vector<Eigen::MatrixX3d> load_lanes_txt(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("load_lanes_txt: cannot open '" + path + "'");

    int num_lanes = 0;
    if (!(f >> num_lanes) || num_lanes < 0)
        throw std::runtime_error("load_lanes_txt: invalid lane count in '" + path + "'");

    std::vector<Eigen::MatrixX3d> lanes;
    lanes.reserve(num_lanes);

    for (int li = 0; li < num_lanes; ++li) {
        int num_points = 0;
        if (!(f >> num_points) || num_points < 0)
            throw std::runtime_error("load_lanes_txt: invalid point count for lane " +
                                     std::to_string(li));

        Eigen::MatrixX3d lane(num_points, 3);
        for (int pi = 0; pi < num_points; ++pi) {
            double x, y, z;
            if (!(f >> x >> y >> z))
                throw std::runtime_error(
                    "load_lanes_txt: unexpected EOF at lane " + std::to_string(li) +
                    ", point " + std::to_string(pi));
            lane(pi, 0) = x;
            lane(pi, 1) = y;
            lane(pi, 2) = z;
        }
        lanes.push_back(std::move(lane));
    }

    return lanes;   // vector of (N×3) matrices
}