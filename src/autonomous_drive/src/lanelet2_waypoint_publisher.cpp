#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int64_multi_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// Lanelet2 core and projection
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_io/Io.h>
#include <lanelet2_projection/UTM.h>

// Lanelet2 routing and traffic rules
#include <lanelet2_routing/RoutingGraph.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <proj.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <cmath>
#include <unordered_set>

class ProjProjector : public lanelet::Projector {
public:
  explicit ProjProjector(const std::string& proj_str, const lanelet::Origin& origin)
  : lanelet::Projector(origin)
  {
    ctx_ = proj_context_create();
    if (!ctx_) {
      throw std::runtime_error("Failed to create PROJ context");
    }
    
    PJ* P = proj_create_crs_to_crs(ctx_, "EPSG:4326", proj_str.c_str(), nullptr);
    if (!P) {
      proj_context_destroy(ctx_);
      throw std::runtime_error("Failed to create PROJ transformation from EPSG:4326 to " + proj_str);
    }
    
    P_norm_ = proj_normalize_for_visualization(ctx_, P);
    proj_destroy(P);
    if (!P_norm_) {
      proj_context_destroy(ctx_);
      throw std::runtime_error("Failed to normalize PROJ transformation for visualization");
    }
  }
  
  ~ProjProjector() noexcept override {
    if (P_norm_) {
      proj_destroy(P_norm_);
    }
    if (ctx_) {
      proj_context_destroy(ctx_);
    }
  }
  
  ProjProjector(const ProjProjector&) = delete;
  ProjProjector& operator=(const ProjProjector&) = delete;
  
  ProjProjector(ProjProjector&& other) noexcept : lanelet::Projector(std::move(other)) {
    ctx_ = other.ctx_;
    P_norm_ = other.P_norm_;
    other.ctx_ = nullptr;
    other.P_norm_ = nullptr;
  }
  
  ProjProjector& operator=(ProjProjector&& other) noexcept {
    if (this != &other) {
      if (P_norm_) proj_destroy(P_norm_);
      if (ctx_) proj_context_destroy(ctx_);
      lanelet::Projector::operator=(std::move(other));
      ctx_ = other.ctx_;
      P_norm_ = other.P_norm_;
      other.ctx_ = nullptr;
      other.P_norm_ = nullptr;
    }
    return *this;
  }

  lanelet::BasicPoint3d forward(const lanelet::GPSPoint& gps) const override {
    PJ_COORD input = proj_coord(gps.lon, gps.lat, gps.ele, 0.0);
    PJ_COORD output = proj_trans(P_norm_, PJ_FWD, input);
    return {output.xyz.x, output.xyz.y, output.xyz.z};
  }

  lanelet::GPSPoint reverse(const lanelet::BasicPoint3d& p) const override {
    PJ_COORD input = proj_coord(p.x(), p.y(), p.z(), 0.0);
    PJ_COORD output = proj_trans(P_norm_, PJ_INV, input);
    return {output.xyz.y, output.xyz.x, output.xyz.z};
  }

private:
  PJ_CONTEXT* ctx_{nullptr};
  PJ* P_norm_{nullptr};
};

class Lanelet2WaypointPublisher : public rclcpp::Node {
public:
  Lanelet2WaypointPublisher() : Node("lanelet2_waypoint_publisher") {
    // Declare parameters
    this->declare_parameter<std::string>("map_file", "/home/marvin/Webots/map4_sumo_to_lanelet.osm");
    this->declare_parameter<double>("origin_lat", 0.0);
    this->declare_parameter<double>("origin_lon", 0.0);
    this->declare_parameter<std::string>("proj_parameter", "+proj=tmerc +lat_0=25.01291 +lon_0=121.46627 +datum=WGS84 +ellps=WGS84 +units=m +no_defs");
    this->declare_parameter<std::string>("frame_id", "map");
    this->declare_parameter<std::vector<int64_t>>("default_lanelet_ids", std::vector<int64_t>{});

    // Get parameters
    map_file_ = this->get_parameter("map_file").as_string();
    origin_lat_ = this->get_parameter("origin_lat").as_double();
    origin_lon_ = this->get_parameter("origin_lon").as_double();
    proj_parameter_ = this->get_parameter("proj_parameter").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
    auto default_ids = this->get_parameter("default_lanelet_ids").as_integer_array();

    // Load Map & build Routing Graph
    if (!loadMapAndGraph()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to load map/routing graph.");
      return;
    }

    // Subscriptions and Publishers
    route_sub_ = this->create_subscription<std_msgs::msg::Int64MultiArray>(
      "/lanelet_route", 10,
      std::bind(&Lanelet2WaypointPublisher::routeCallback, this, std::placeholders::_1));

    path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/centerline_waypoints", rclcpp::QoS(1).transient_local());
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/centerline_waypoints_markers", rclcpp::QoS(1).transient_local());

    // Process default sequence if provided
    if (!default_ids.empty()) {
      RCLCPP_INFO(this->get_logger(), "Processing default lanelet sequence of size %zu.", default_ids.size());
      processRoute(default_ids);
    } else {
      RCLCPP_WARN(this->get_logger(), "Default lanelet sequence is empty.");
    }
  }

private:
  bool loadMapAndGraph() {
    RCLCPP_INFO(this->get_logger(), "Loading Lanelet2 map: %s", map_file_.c_str());
    try {
      lanelet::GPSPoint gps_origin{origin_lat_, origin_lon_, 0.0};
      lanelet::Origin origin(gps_origin);
      ProjProjector projector(proj_parameter_, origin);

      lanelet::ErrorMessages errors;
      map_ = lanelet::load(map_file_, projector, &errors);

      for (const auto & e : errors) {
        RCLCPP_WARN(this->get_logger(), "Map load warning: %s", e.c_str());
      }

      RCLCPP_INFO(this->get_logger(),
        "Map loaded — lanelets: %zu  linestrings: %zu",
        map_->laneletLayer.size(),
        map_->lineStringLayer.size());

      // Create traffic rules (standard vehicle participant)
      traffic_rules_ = lanelet::traffic_rules::TrafficRulesFactory::create(
        lanelet::Locations::Germany, lanelet::Participants::Vehicle);

      // Build routing graph
      routing_graph_ = lanelet::routing::RoutingGraph::build(*map_, *traffic_rules_);
      RCLCPP_INFO(this->get_logger(), "Routing graph successfully built.");
      return true;
    }
    catch (const std::exception & ex) {
      RCLCPP_ERROR(this->get_logger(), "Exception loading map/graph: %s", ex.what());
      return false;
    }
  }

  void routeCallback(const std_msgs::msg::Int64MultiArray::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received lanelet ID sequence on /lanelet_route of size %zu.", msg->data.size());
    processRoute(msg->data);
  }

  void processRoute(const std::vector<int64_t> & ids) {
    if (ids.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received sequence is empty. Cannot process waypoints.");
      return;
    }

    // 1. Resolve Lanelet IDs
    std::vector<lanelet::ConstLanelet> lanelets;
    for (int64_t id : ids) {
      if (!map_->laneletLayer.exists(id)) {
        RCLCPP_ERROR(this->get_logger(), "Lanelet ID %ld does not exist in the map!", id);
        return;
      }
      lanelets.push_back(map_->laneletLayer.get(id));
    }

    // 2. Validate routing connectivity
    bool route_valid = true;
    for (size_t i = 0; i + 1 < lanelets.size(); ++i) {
      auto relation = routing_graph_->routingRelation(lanelets[i], lanelets[i+1]);
      if (!relation ||
          (*relation != lanelet::routing::RelationType::Successor &&
           *relation != lanelet::routing::RelationType::Left &&
           *relation != lanelet::routing::RelationType::Right)) {
        RCLCPP_ERROR(this->get_logger(), "Invalid route transition from lanelet %ld to %ld!", 
                     lanelets[i].id(), lanelets[i+1].id());
        if (relation) {
          RCLCPP_ERROR(this->get_logger(), "Relation type found: %d (but successor, left, or right was expected)", 
                       static_cast<int>(*relation));
        } else {
          RCLCPP_ERROR(this->get_logger(), "No relation found between them.");
        }
        route_valid = false;
        break;
      }
    }

    if (!route_valid) {
      RCLCPP_ERROR(this->get_logger(), "Provided sequence does not form a valid route. Aborting waypoint publication.");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Lanelet sequence forms a valid route. Generating waypoints.");

    // 3. Extract and concatenate centerline points
    std::vector<geometry_msgs::msg::Point> raw_points;
    for (const auto & ll : lanelets) {
      const auto & centerline = ll.centerline();
      for (const auto & pt : centerline) {
        geometry_msgs::msg::Point gp;
        gp.x = pt.x();
        gp.y = pt.y();
        gp.z = pt.z();
        raw_points.push_back(gp);
      }
    }

    // 4. Deduplicate close/overlapping consecutive points
    std::vector<geometry_msgs::msg::Point> unique_points;
    for (const auto & pt : raw_points) {
      if (unique_points.empty()) {
        unique_points.push_back(pt);
      } else {
        const auto & last = unique_points.back();
        double dx = pt.x - last.x;
        double dy = pt.y - last.y;
        double dz = pt.z - last.z;
        double dist = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (dist > 0.05) { // 5 cm threshold
          unique_points.push_back(pt);
        }
      }
    }

    if (unique_points.size() < 2) {
      RCLCPP_WARN(this->get_logger(), "Not enough waypoints to compute directions (less than 2 unique points).");
      return;
    }

    // 5. Build nav_msgs::msg::Path with 3D orientations (pointing to next waypoint)
    nav_msgs::msg::Path path;
    path.header.frame_id = frame_id_;
    path.header.stamp = this->now();

    for (size_t i = 0; i < unique_points.size(); ++i) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position = unique_points[i];

      double dx = 0.0, dy = 0.0, dz = 0.0;
      if (i + 1 < unique_points.size()) {
        dx = unique_points[i+1].x - unique_points[i].x;
        dy = unique_points[i+1].y - unique_points[i].y;
        dz = unique_points[i+1].z - unique_points[i].z;
      } else if (i > 0) {
        dx = unique_points[i].x - unique_points[i-1].x;
        dy = unique_points[i].y - unique_points[i-1].y;
        dz = unique_points[i].z - unique_points[i-1].z;
      }

      Eigen::Vector3d dir(dx, dy, dz);
      if (dir.norm() > 1e-4) {
        dir.normalize();
        Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitX(), dir);
        pose.pose.orientation.x = q.x();
        pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z();
        pose.pose.orientation.w = q.w();
      } else {
        pose.pose.orientation.w = 1.0;
      }
      path.poses.push_back(pose);
    }

    // Publish path
    path_pub_->publish(path);
    RCLCPP_INFO(this->get_logger(), "Published %zu waypoints to /centerline_waypoints.", path.poses.size());

    // 6. Build and publish MarkerArray for visualization
    visualization_msgs::msg::MarkerArray marker_array;

    // Line strip
    visualization_msgs::msg::Marker line_marker;
    line_marker.header = path.header;
    line_marker.ns = "centerline_line";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.15;
    line_marker.color.r = 0.0f;
    line_marker.color.g = 0.8f;
    line_marker.color.b = 1.0f;
    line_marker.color.a = 0.8f;
    line_marker.pose.orientation.w = 1.0;

    int sphere_id = 1;
    for (const auto & pose : path.poses) {
      line_marker.points.push_back(pose.pose.position);

      visualization_msgs::msg::Marker sphere_marker;
      sphere_marker.header = path.header;
      sphere_marker.ns = "centerline_points";
      sphere_marker.id = sphere_id++;
      sphere_marker.type = visualization_msgs::msg::Marker::SPHERE;
      sphere_marker.action = visualization_msgs::msg::Marker::ADD;
      sphere_marker.pose = pose.pose;
      sphere_marker.scale.x = 0.3;
      sphere_marker.scale.y = 0.3;
      sphere_marker.scale.z = 0.3;
      sphere_marker.color.r = 0.0f;
      sphere_marker.color.g = 1.0f;
      sphere_marker.color.b = 0.0f;
      sphere_marker.color.a = 1.0f;
      marker_array.markers.push_back(sphere_marker);
    }

    if (!line_marker.points.empty()) {
      marker_array.markers.push_back(line_marker);
    }

    marker_pub_->publish(marker_array);
  }

  // Parameters
  std::string map_file_;
  double origin_lat_;
  double origin_lon_;
  std::string proj_parameter_;
  std::string frame_id_;

  // Lanelet2 objects
  lanelet::LaneletMapPtr map_;
  lanelet::traffic_rules::TrafficRulesPtr traffic_rules_;
  lanelet::routing::RoutingGraphUPtr routing_graph_;

  // Subscriptions and Publishers
  rclcpp::Subscription<std_msgs::msg::Int64MultiArray>::SharedPtr route_sub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Lanelet2WaypointPublisher>());
  rclcpp::shutdown();
  return 0;
}
