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

// ---------------------------------------------------------------------------
// Cubic Hermite spline interpolation helper
//
//   H(t) = (2t³-3t²+1)·p0 + (t³-2t²+t)·m0 + (-2t³+3t²)·p1 + (t³-t²)·m1
//
// p0, p1 : endpoint positions
// m0, m1 : endpoint tangents (scaled to approximate arc length)
// n_pts  : number of output points (inclusive of both endpoints)
// ---------------------------------------------------------------------------
static std::vector<Eigen::Vector3d> cubicHermiteSpline(
  const Eigen::Vector3d& p0, const Eigen::Vector3d& m0,
  const Eigen::Vector3d& p1, const Eigen::Vector3d& m1,
  int n_pts)
{
  std::vector<Eigen::Vector3d> result;
  result.reserve(n_pts);
  for (int i = 0; i < n_pts; ++i) {
    double t  = static_cast<double>(i) / static_cast<double>(n_pts - 1);
    double t2 = t * t;
    double t3 = t2 * t;
    double h00 =  2*t3 - 3*t2 + 1;
    double h10 =    t3 - 2*t2 + t;
    double h01 = -2*t3 + 3*t2;
    double h11 =    t3 -   t2;
    result.push_back(h00*p0 + h10*m0 + h01*p1 + h11*m1);
  }
  return result;
}

// Extract points from a lanelet centerline as Eigen vectors
static std::vector<Eigen::Vector3d> centerlinePoints(const lanelet::ConstLanelet& ll)
{
  std::vector<Eigen::Vector3d> pts;
  for (const auto& pt : ll.centerline()) {
    pts.emplace_back(pt.x(), pt.y(), pt.z());
  }
  return pts;
}

// Compute cumulative arc-length from start to each vertex
static std::vector<double> arcLengths(const std::vector<Eigen::Vector3d>& pts)
{
  std::vector<double> s(pts.size(), 0.0);
  for (size_t i = 1; i < pts.size(); ++i) {
    s[i] = s[i-1] + (pts[i] - pts[i-1]).norm();
  }
  return s;
}

// Sample a polyline at fraction [0,1] of its total arc length.
// Returns {interpolated position, local unit tangent}.
static std::pair<Eigen::Vector3d, Eigen::Vector3d>
samplePolyline(const std::vector<Eigen::Vector3d>& pts,
               const std::vector<double>& s,
               double fraction)
{
  if (pts.size() == 1) {
    return {pts[0], Eigen::Vector3d::UnitX()};
  }
  double target = std::max(0.0, std::min(s.back(), fraction * s.back()));
  for (size_t i = 1; i < pts.size(); ++i) {
    if (s[i] >= target - 1e-9) {
      double seg_len = s[i] - s[i-1];
      double local_t = (seg_len > 1e-9) ? (target - s[i-1]) / seg_len : 0.0;
      Eigen::Vector3d pos = pts[i-1] + local_t * (pts[i] - pts[i-1]);
      Eigen::Vector3d tan = pts[i] - pts[i-1];
      if (tan.norm() > 1e-9) tan.normalize();
      return {pos, tan};
    }
  }
  // Fallback: end
  size_t n = pts.size();
  Eigen::Vector3d tan = (pts[n-1] - pts[n-2]);
  if (tan.norm() > 1e-9) tan.normalize();
  return {pts.back(), tan};
}

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

    // lane_change_spline_ratio (s) ∈ [0, 0.5):
    //   Fraction of total traversal consumed by each spline bridge.
    //   At each lane-change boundary, s/2 is cut from the end of the departing
    //   lane and s/2 from the start of the arriving lane.
    //   Example with N=3 lanes and s=0.10:
    //     Each lane owns 1/3 of the route.  Slices (with splines removed):
    //     Lane 0: [  0% .. 28.3%]  spline  Lane 1: [38.3% .. 61.7%]  spline  Lane 2: [71.7% .. 100%]
    this->declare_parameter<double>("lane_change_spline_ratio", 0.10);

    // Number of interpolated points generated for each spline bridge.
    this->declare_parameter<int>("lane_change_spline_points", 20);

    // Get parameters
    map_file_ = this->get_parameter("map_file").as_string();
    origin_lat_ = this->get_parameter("origin_lat").as_double();
    origin_lon_ = this->get_parameter("origin_lon").as_double();
    proj_parameter_ = this->get_parameter("proj_parameter").as_string();
    frame_id_ = this->get_parameter("frame_id").as_string();
    auto default_ids = this->get_parameter("default_lanelet_ids").as_integer_array();
    lane_change_spline_ratio_  = this->get_parameter("lane_change_spline_ratio").as_double();
    lane_change_spline_points_ = this->get_parameter("lane_change_spline_points").as_int();
    // Clamp: s must be < 1/N for all N >= 2, so cap at 0.49 (further clamped per-group)
    lane_change_spline_ratio_  = std::max(0.0, std::min(0.49, lane_change_spline_ratio_));
    lane_change_spline_points_ = std::max(3, lane_change_spline_points_);

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
    //    Successor → normal following lane
    //    Left / Right → lane change (accepted)
    bool route_valid = true;
    for (size_t i = 0; i + 1 < lanelets.size(); ++i) {
      auto relation = routing_graph_->routingRelation(lanelets[i], lanelets[i+1]);
      bool ok = relation &&
                (*relation == lanelet::routing::RelationType::Successor ||
                 *relation == lanelet::routing::RelationType::Left      ||
                 *relation == lanelet::routing::RelationType::Right);
      if (!ok) {
        RCLCPP_ERROR(this->get_logger(), "Invalid route transition from lanelet %ld to %ld!",
                     lanelets[i].id(), lanelets[i+1].id());
        if (relation) {
          RCLCPP_ERROR(this->get_logger(),
            "Relation type found: %d (expected Successor, Left, or Right)",
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
    //
    //  Look ahead from the current lane to collect all consecutive Left/Right
    //  adjacent lanelets into a "LC group", stopping as soon as the next
    //  relation is Successor (or we run out of lanelets).
    //  The full group is processed together by processLaneChangeGroup().
    //  Single-lane (Successor) segments are appended in full.
    std::vector<geometry_msgs::msg::Point> raw_points;

    size_t lane_idx = 0;
    while (lane_idx < lanelets.size()) {
      // Scan ahead to measure the LC group that starts here
      size_t group_size = 1;
      while (lane_idx + group_size < lanelets.size()) {
        auto rel = routing_graph_->routingRelation(
          lanelets[lane_idx + group_size - 1],
          lanelets[lane_idx + group_size]);
        if (rel &&
            (*rel == lanelet::routing::RelationType::Left ||
             *rel == lanelet::routing::RelationType::Right)) {
          ++group_size;
        } else {
          break;
        }
      }

      if (group_size == 1) {
        // Single lane (Successor or last): append full centerline
        for (const auto& pt : lanelets[lane_idx].centerline()) {
          geometry_msgs::msg::Point gp;
          gp.x = pt.x(); gp.y = pt.y(); gp.z = pt.z();
          raw_points.push_back(gp);
        }
      } else {
        // LC group: equal-split with spline bridges at boundaries
        processLaneChangeGroup(lanelets, lane_idx, group_size, raw_points);
      }
      lane_idx += group_size;
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

    for (size_t k = 0; k < unique_points.size(); ++k) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position = unique_points[k];

      double dx = 0.0, dy = 0.0, dz = 0.0;
      if (k + 1 < unique_points.size()) {
        dx = unique_points[k+1].x - unique_points[k].x;
        dy = unique_points[k+1].y - unique_points[k].y;
        dz = unique_points[k+1].z - unique_points[k].z;
      } else if (k > 0) {
        dx = unique_points[k].x - unique_points[k-1].x;
        dy = unique_points[k].y - unique_points[k-1].y;
        dz = unique_points[k].z - unique_points[k-1].z;
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

  // -------------------------------------------------------------------------
  // Process a group of N consecutive Left/Right adjacent lanelets as a single
  // smooth lane-change trajectory.
  //
  // Each of the N lanes owns an equal 1/N fraction of the traversal.
  // At each boundary between lane i and lane i+1 a cubic Hermite spline bridge
  // spans 'lane_change_spline_ratio_' (s) of the route: s/2 is removed from
  // the end of the departing lane and s/2 from the start of the arriving lane.
  //
  // Layout for N=3, s=10%  (seg = 33.3%, hs = 5%):
  //   Lane 0 : [  0%  ..  28.3%]  <- seg - hs
  //   Spline : [ 28.3% ..  38.3%] <- s
  //   Lane 1 : [ 38.3% ..  61.7%] <- seg - 2*hs  (inner lane)
  //   Spline : [ 61.7% ..  71.7%] <- s
  //   Lane 2 : [ 71.7% .. 100% ]  <- seg - hs
  // -------------------------------------------------------------------------
  void processLaneChangeGroup(
    const std::vector<lanelet::ConstLanelet>& all_lanelets,
    size_t start_idx,
    size_t N,
    std::vector<geometry_msgs::msg::Point>& raw_points) const
  {
    // Safety: if s >= 2/N the inner-lane slices would invert; cap at (1/N - epsilon)
    const double seg = 1.0 / static_cast<double>(N);
    const double s   = std::min(lane_change_spline_ratio_, seg - 1e-3);
    const double hs  = s / 2.0;

    auto push_pt = [&](const Eigen::Vector3d& p) {
      geometry_msgs::msg::Point gp;
      gp.x = p.x(); gp.y = p.y(); gp.z = p.z();
      raw_points.push_back(gp);
    };

    for (size_t i = 0; i < N; ++i) {
      const auto& ll  = all_lanelets[start_idx + i];
      auto pts = centerlinePoints(ll);
      auto sv  = arcLengths(pts);

      // Fraction boundaries for this lane's slice
      // First lane: [0, seg - hs]   Inner lanes: [i*seg + hs, (i+1)*seg - hs]
      // Last  lane: [(N-1)*seg + hs, 1.0]
      double sf = (i == 0)     ? 0.0            : i * seg + hs;
      double ef = (i == N - 1) ? 1.0            : (i + 1) * seg - hs;
      sf = std::max(0.0, std::min(1.0, sf));
      ef = std::max(sf + 1e-6, std::min(1.0, ef));

      // Interpolated endpoints and tangents
      auto [p_sf, _ts] = samplePolyline(pts, sv, sf);
      auto [p_ef, tan_ef] = samplePolyline(pts, sv, ef);

      // --- Lane slice ---
      // For i > 0: the slice's start point was already placed by the previous
      // bridge's last point (bridge ends at p_sf).  Skip it to avoid duplicates.
      if (i == 0) {
        push_pt(p_sf);
      }
      // Interior vertices strictly between sf and ef
      double sd = sf * sv.back();
      double ed = ef * sv.back();
      for (size_t j = 0; j < pts.size(); ++j) {
        if (sv[j] > sd + 1e-9 && sv[j] < ed - 1e-9) {
          push_pt(pts[j]);
        }
      }
      push_pt(p_ef);  // end of slice / start of bridge

      // --- Spline bridge to the next lane ---
      if (i + 1 < N) {
        const auto& nxt_ll = all_lanelets[start_idx + i + 1];
        auto nxt_pts = centerlinePoints(nxt_ll);
        auto nxt_sv  = arcLengths(nxt_pts);

        // Next lane's slice starts at (i+1)*seg + hs
        double nxt_sf = (i + 1) * seg + hs;
        nxt_sf = std::max(0.0, std::min(1.0, nxt_sf));

        auto [p_nxt, tan_nxt] = samplePolyline(nxt_pts, nxt_sv, nxt_sf);

        double chord = (p_nxt - p_ef).norm();
        auto bridge = cubicHermiteSpline(
          p_ef,  tan_ef  * chord,
          p_nxt, tan_nxt * chord,
          lane_change_spline_points_);

        // bridge[0] == p_ef (already pushed); add bridge[1 .. end]
        // bridge.back() == p_nxt == next lane's sf → next iteration skips it
        for (size_t k = 1; k < bridge.size(); ++k) {
          push_pt(bridge[k]);
        }
      }
    }

    RCLCPP_INFO(this->get_logger(),
      "LC group: %zu lanes | seg=%.1f%% spline=%.1f%% (±%.1f%% each side) "
      "| lanelets %ld .. %ld",
      N, seg * 100.0, s * 100.0, hs * 100.0,
      all_lanelets[start_idx].id(),
      all_lanelets[start_idx + N - 1].id());
  }

  // Parameters
  std::string map_file_;
  double origin_lat_;
  double origin_lon_;
  std::string proj_parameter_;
  std::string frame_id_;

  // Lane-change parameters
  double lane_change_spline_ratio_{0.10}; ///< s: fraction of each segment boundary consumed by the spline bridge
  int    lane_change_spline_points_{20};  ///< number of points in each cubic Hermite bridge

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
