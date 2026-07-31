#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <memory>
#include <stdexcept>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

struct GPSCoord {
    double lat{0.0}; // Latitude in degrees
    double lon{0.0}; // Longitude in degrees
    double alt{0.0}; // Altitude in meters
};

struct RPYCoord {
    double roll{0.0};  // Roll in degrees (around X)
    double pitch{0.0}; // Pitch in degrees (around Y)
    double yaw{0.0};   // Yaw in degrees (around Z / Heading)
};

struct ECEFCoord {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct ENUCoord {
    double east{0.0};
    double north{0.0};
    double up{0.0};
};

// Convert Geodetic (WGS84) to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates
ECEFCoord geodeticToECEF(const GPSCoord& gps) {
    constexpr double a = 6378137.0;           // WGS84 semi-major axis in meters
    constexpr double f = 1.0 / 298.257223563;  // WGS84 flattening
    constexpr double e2 = 2.0 * f - f * f;    // First eccentricity squared

    double lat_rad = gps.lat * M_PI / 180.0;
    double lon_rad = gps.lon * M_PI / 180.0;

    double sin_lat = std::sin(lat_rad);
    double cos_lat = std::cos(lat_rad);
    double sin_lon = std::sin(lon_rad);
    double cos_lon = std::cos(lon_rad);

    double N = a / std::sqrt(1.0 - e2 * sin_lat * sin_lat);

    ECEFCoord ecef;
    ecef.x = (N + gps.alt) * cos_lat * cos_lon;
    ecef.y = (N + gps.alt) * cos_lat * sin_lon;
    ecef.z = (N * (1.0 - e2) + gps.alt) * sin_lat;
    return ecef;
}

// Convert ECEF coordinate difference to ENU relative to ref_gps
ENUCoord gpsToENU(const GPSCoord& target_gps, const GPSCoord& ref_gps) {
    ECEFCoord target_ecef = geodeticToECEF(target_gps);
    ECEFCoord ref_ecef = geodeticToECEF(ref_gps);

    double dx = target_ecef.x - ref_ecef.x;
    double dy = target_ecef.y - ref_ecef.y;
    double dz = target_ecef.z - ref_ecef.z;

    double lat_rad = ref_gps.lat * M_PI / 180.0;
    double lon_rad = ref_gps.lon * M_PI / 180.0;

    double sin_lat = std::sin(lat_rad);
    double cos_lat = std::cos(lat_rad);
    double sin_lon = std::sin(lon_rad);
    double cos_lon = std::cos(lon_rad);

    ENUCoord enu;
    enu.east  = -sin_lon * dx + cos_lon * dy;
    enu.north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz;
    enu.up    =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz;
    return enu;
}

// Convert Roll, Pitch, Yaw angles (in degrees) to a 3x3 Rotation Matrix
Eigen::Matrix3f rpyToRotationMatrix(double roll_deg, double pitch_deg, double yaw_deg) {
    float roll  = static_cast<float>(roll_deg  * M_PI / 180.0);
    float pitch = static_cast<float>(pitch_deg * M_PI / 180.0);
    float yaw   = static_cast<float>(yaw_deg   * M_PI / 180.0);

    Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());

    Eigen::Quaternionf q = yawAngle * pitchAngle * rollAngle;
    return q.toRotationMatrix();
}

void printUsage(const char* prog_name) {
    std::cout << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --map1 <file.pcd>           Path to Map 1 PCD (Reference Map)\n"
              << "  --gps1 <v0> <v1> [v2]       GPS coordinate for Map 1 initial position (0,0)\n"
              << "  --map2 <file.pcd>           Path to Map 2 PCD\n"
              << "  --gps2 <v0> <v1> [v2]       GPS coordinate for Map 2 initial position (0,0)\n"
              << "  --quat1 <x> <y> <z> <w>     IMU Orientation Quaternion for Map 1 (x y z w)\n"
              << "  --quat2 <x> <y> <z> <w>     IMU Orientation Quaternion for Map 2 (x y z w)\n"
              << "  --yaw1 <deg>                Initial vehicle yaw/heading angle for Map 1 (deg)\n"
              << "  --yaw2 <deg>                Initial vehicle yaw/heading angle for Map 2 (deg)\n"
              << "  --rpy1 <r> <p> <y>          Initial Roll, Pitch, Yaw angles for Map 1 (deg)\n"
              << "  --rpy2 <r> <p> <y>          Initial Roll, Pitch, Yaw angles for Map 2 (deg)\n"
              << "  --out2 <file.pcd>           Output path for aligned Map 2 (default: map2_aligned.pcd)\n"
              << "  --merged <file.pcd>         Output path for merged map (default: merged_map.pcd)\n"
              << "  --order <lat_lon_alt|lat_alt_lon>\n"
              << "                              GPS array order (default: lat_lon_alt)\n"
              << "                              Use 'lat_alt_lon' if passing raw Webots wb_gps_get_values array\n"
              << "  --help                      Show this help message\n\n"
              << "Example (Quaternion from ROS IMU):\n"
              << "  " << prog_name << " \\\n"
              << "    --map1 session1_map/GlobalMap.pcd --gps1 25.012562 121.466838 1.557820 \\\n"
              << "    --quat1 -0.0007625 -0.0003750 -0.5190791 0.8546692 \\\n"
              << "    --map2 session2_map/GlobalMap.pcd --gps2 25.012910 121.466271 0.000000 \\\n"
              << "    --quat2 0.0005000 -0.0009000 -0.5203921 0.8538531 \\\n"
              << "    --out2 map2_aligned.pcd --merged map_merged.pcd\n";
}

GPSCoord parseGPS(double v0, double v1, double v2, const std::string& order) {
    GPSCoord gps;
    if (order == "lat_alt_lon") { // Raw Webots wb_gps_get_values order
        gps.lat = v0;
        gps.alt = v1;
        gps.lon = v2;
    } else { // Standard lat_lon_alt
        gps.lat = v0;
        gps.lon = v1;
        gps.alt = v2;
    }
    return gps;
}

int main(int argc, char** argv) {
    std::string map1_path;
    std::string map2_path;
    std::string out2_path = "map2_aligned.pcd";
    std::string merged_path = "merged_map.pcd";
    std::string order = "lat_lon_alt";

    GPSCoord gps1;
    GPSCoord gps2;
    bool has_gps1 = false;
    bool has_gps2 = false;

    RPYCoord rpy1{0.0, 0.0, 0.0};
    RPYCoord rpy2{0.0, 0.0, 0.0};
    bool has_rpy1 = false;
    bool has_rpy2 = false;

    Eigen::Quaternionf quat1 = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf quat2 = Eigen::Quaternionf::Identity();
    bool has_quat1 = false;
    bool has_quat2 = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--map1" && i + 1 < argc) {
            map1_path = argv[++i];
        } else if (arg == "--map2" && i + 1 < argc) {
            map2_path = argv[++i];
        } else if (arg == "--out2" && i + 1 < argc) {
            out2_path = argv[++i];
        } else if (arg == "--merged" && i + 1 < argc) {
            merged_path = argv[++i];
        } else if (arg == "--order" && i + 1 < argc) {
            order = argv[++i];
        } else if (arg == "--yaw1" && i + 1 < argc) {
            rpy1.yaw = std::stod(argv[++i]);
            has_rpy1 = true;
        } else if (arg == "--yaw2" && i + 1 < argc) {
            rpy2.yaw = std::stod(argv[++i]);
            has_rpy2 = true;
        } else if (arg == "--rpy1" && i + 3 < argc) {
            rpy1.roll  = std::stod(argv[++i]);
            rpy1.pitch = std::stod(argv[++i]);
            rpy1.yaw   = std::stod(argv[++i]);
            has_rpy1 = true;
        } else if (arg == "--rpy2" && i + 3 < argc) {
            rpy2.roll  = std::stod(argv[++i]);
            rpy2.pitch = std::stod(argv[++i]);
            rpy2.yaw   = std::stod(argv[++i]);
            has_rpy2 = true;
        } else if (arg == "--quat1" && i + 4 < argc) {
            float x = std::stof(argv[++i]);
            float y = std::stof(argv[++i]);
            float z = std::stof(argv[++i]);
            float w = std::stof(argv[++i]);
            quat1 = Eigen::Quaternionf(w, x, y, z).normalized();
            has_quat1 = true;
        } else if (arg == "--quat2" && i + 4 < argc) {
            float x = std::stof(argv[++i]);
            float y = std::stof(argv[++i]);
            float z = std::stof(argv[++i]);
            float w = std::stof(argv[++i]);
            quat2 = Eigen::Quaternionf(w, x, y, z).normalized();
            has_quat2 = true;
        } else if (arg == "--gps1" && i + 2 < argc) {
            double v0 = std::stod(argv[++i]);
            double v1 = std::stod(argv[++i]);
            double v2 = 0.0;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                v2 = std::stod(argv[++i]);
            }
            gps1 = parseGPS(v0, v1, v2, order);
            has_gps1 = true;
        } else if (arg == "--gps2" && i + 2 < argc) {
            double v0 = std::stod(argv[++i]);
            double v1 = std::stod(argv[++i]);
            double v2 = 0.0;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                v2 = std::stod(argv[++i]);
            }
            gps2 = parseGPS(v0, v1, v2, order);
            has_gps2 = true;
        }
    }

    if (map1_path.empty() || map2_path.empty() || !has_gps1 || !has_gps2) {
        std::cerr << "Error: Missing required arguments.\n\n";
        printUsage(argv[0]);
        return 1;
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "==================================================\n";
    std::cout << "PCD Map Alignment Tool (GPS WGS84 & IMU Orientation)\n";
    std::cout << "==================================================\n";
    std::cout << "Map 1 PCD : " << map1_path << "\n";
    std::cout << "Map 1 GPS : Lat = " << gps1.lat << ", Lon = " << gps1.lon << ", Alt = " << gps1.alt << " m\n";
    std::cout << "Map 2 PCD : " << map2_path << "\n";
    std::cout << "Map 2 GPS : Lat = " << gps2.lat << ", Lon = " << gps2.lon << ", Alt = " << gps2.alt << " m\n";
    std::cout << "GPS Order : " << order << "\n";

    Eigen::Matrix3f R1 = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f R2 = Eigen::Matrix3f::Identity();

    if (has_quat1) {
        R1 = quat1.toRotationMatrix();
        std::cout << "Map 1 Quat: x=" << quat1.x() << ", y=" << quat1.y() << ", z=" << quat1.z() << ", w=" << quat1.w() << "\n";
    } else if (has_rpy1) {
        R1 = rpyToRotationMatrix(rpy1.roll, rpy1.pitch, rpy1.yaw);
        std::cout << "Map 1 RPY : Roll=" << rpy1.roll << " deg, Pitch=" << rpy1.pitch << " deg, Yaw=" << rpy1.yaw << " deg\n";
    }

    if (has_quat2) {
        R2 = quat2.toRotationMatrix();
        std::cout << "Map 2 Quat: x=" << quat2.x() << ", y=" << quat2.y() << ", z=" << quat2.z() << ", w=" << quat2.w() << "\n";
    } else if (has_rpy2) {
        R2 = rpyToRotationMatrix(rpy2.roll, rpy2.pitch, rpy2.yaw);
        std::cout << "Map 2 RPY : Roll=" << rpy2.roll << " deg, Pitch=" << rpy2.pitch << " deg, Yaw=" << rpy2.yaw << " deg\n";
    }

    std::cout << "--------------------------------------------------\n";

    // Compute relative ENU displacement of Map 2 origin relative to Map 1 origin
    ENUCoord enu = gpsToENU(gps2, gps1);

    std::cout << "[ENU Translation Offset (Map 2 origin relative to Map 1 origin)]\n";
    std::cout << "  East  (X): " << std::setprecision(4) << enu.east << " m\n";
    std::cout << "  North (Y): " << enu.north << " m\n";
    std::cout << "  Up    (Z): " << enu.up << " m\n";
    std::cout << "--------------------------------------------------\n";

    // Relative Rotation Matrix to transform Map 2 points into Map 1's orientation frame:
    // R_rel = R1^-1 * R2 = R1^T * R2
    Eigen::Matrix3f R_rel = R1.transpose() * R2;

    // Load PCD files
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZI>());

    std::cout << "Loading Map 1 PCD..." << std::flush;
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map1_path, *cloud1) == -1) {
        std::cerr << "\nFailed to load Map 1: " << map1_path << std::endl;
        return 1;
    }
    std::cout << " Done (" << cloud1->points.size() << " points).\n";

    std::cout << "Loading Map 2 PCD..." << std::flush;
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(map2_path, *cloud2) == -1) {
        std::cerr << "\nFailed to load Map 2: " << map2_path << std::endl;
        return 1;
    }
    std::cout << " Done (" << cloud2->points.size() << " points).\n";

    // Build 4x4 Rigid Transformation Matrix (Rotation + Translation)
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.linear() = R_rel;
    transform.translation() << static_cast<float>(enu.east),
                               static_cast<float>(enu.north),
                               static_cast<float>(enu.up);

    std::cout << "\nTransformation Matrix (4x4):\n" << transform.matrix() << "\n\n";

    // Transform Map 2 point cloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud2_aligned(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cloud2, *cloud2_aligned, transform);

    // Merge Map 1 and Map 2 aligned
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_merged(new pcl::PointCloud<pcl::PointXYZI>());
    *cloud_merged = *cloud1 + *cloud2_aligned;

    // Save outputs
    std::cout << "Saving aligned Map 2 to: " << out2_path << "..." << std::flush;
    if (pcl::io::savePCDFileBinaryCompressed(out2_path, *cloud2_aligned) == -1) {
        std::cerr << "\nFailed to save aligned Map 2 to " << out2_path << std::endl;
        return 1;
    }
    std::cout << " Done.\n";

    std::cout << "Saving merged map to: " << merged_path << "..." << std::flush;
    if (pcl::io::savePCDFileBinaryCompressed(merged_path, *cloud_merged) == -1) {
        std::cerr << "\nFailed to save merged map to " << merged_path << std::endl;
        return 1;
    }
    std::cout << " Done.\n";

    std::cout << "==================================================\n";
    std::cout << "Alignment finished successfully!\n";
    std::cout << "  Merged map point count: " << cloud_merged->points.size() << "\n";
    std::cout << "==================================================\n";

    return 0;
}
