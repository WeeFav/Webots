#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/color_rgba.hpp>

// Lanelet2 core headers
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/geometry/Point.h>
#include <lanelet2_core/primitives/Lanelet.h>
#include <lanelet2_core/primitives/LineString.h>
#include <lanelet2_core/primitives/Area.h>

// Lanelet2 I/O and projection
#include <lanelet2_io/Io.h>
#include <lanelet2_io/io_handlers/Factory.h>
#include <lanelet2_projection/UTM.h>

#include <proj.h>

#include <chrono>
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace std::chrono_literals;

// ---------------------------------------------------------------------------
// Helper: build a solid color RGBA message
// ---------------------------------------------------------------------------
std_msgs::msg::ColorRGBA makeColor(float r, float g, float b, float a = 1.0f)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r; c.g = g; c.b = b; c.a = a;
  return c;
}

// ---------------------------------------------------------------------------
// Helper: convert a Lanelet2 3-D point to a geometry_msgs Point
// ---------------------------------------------------------------------------
geometry_msgs::msg::Point toPoint(const lanelet::ConstPoint3d & p)
{
  geometry_msgs::msg::Point gp;
  gp.x = p.x(); gp.y = p.y(); gp.z = p.z();
  return gp;
}

// Helper: parse a string like "[1.0, 2.0, 3.0]" into a vector of doubles
std::vector<double> parseDoubleArray(const std::string & s)
{
  std::vector<double> res;
  std::string clean = s;
  clean.erase(std::remove(clean.begin(), clean.end(), '['), clean.end());
  clean.erase(std::remove(clean.begin(), clean.end(), ']'), clean.end());
  clean.erase(std::remove(clean.begin(), clean.end(), ' '), clean.end());
  
  std::stringstream ss(clean);
  std::string token;
  while (std::getline(ss, token, ',')) {
    if (!token.empty()) {
      try {
        res.push_back(std::stod(token));
      } catch (...) {}
    }
  }
  return res;
}

// Helper: retrieve parameter as double array, parsing string if necessary
std::vector<double> getParameterDoubleArray(rclcpp::Node * node, const std::string & name)
{
  auto param = node->get_parameter(name);
  if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE_ARRAY) {
    return param.as_double_array();
  } else if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING) {
    return parseDoubleArray(param.as_string());
  }
  return std::vector<double>{};
}

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
  
  // Disable copying/assignment because we manage resources manually
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
// Node
// ---------------------------------------------------------------------------
class Lanelet2Visualizer : public rclcpp::Node
{
public:
  Lanelet2Visualizer()
  : Node("lanelet2_visualizer")
  {
    // ---- Parameters -------------------------------------------------------
    this->declare_parameter<std::string>("map_file",   "map.osm");
    this->declare_parameter<std::string>("frame_id",   "map");
    this->declare_parameter<double>("origin_lat",       0.0);
    this->declare_parameter<double>("origin_lon",       0.0);
    this->declare_parameter<std::string>("projParameter", "+proj=tmerc +lat_0=25.01291 +lon_0=121.46627 +datum=WGS84 +ellps=WGS84 +units=m +no_defs");
    this->declare_parameter<double>("publish_rate_hz",  1.0);

    rcl_interfaces::msg::ParameterDescriptor points_desc;
    points_desc.dynamic_typing = true;
    this->declare_parameter("points_x", rclcpp::ParameterValue(std::vector<double>{}), points_desc);
    this->declare_parameter("points_y", rclcpp::ParameterValue(std::vector<double>{}), points_desc);
    this->declare_parameter("points_z", rclcpp::ParameterValue(std::vector<double>{}), points_desc);

    this->declare_parameter<double>("spheres_scale",    0.3);
    this->declare_parameter<double>("spheres_color_r",  1.0);
    this->declare_parameter<double>("spheres_color_g",  0.5);
    this->declare_parameter<double>("spheres_color_b",  0.0);
    this->declare_parameter<double>("spheres_color_a",  1.0);

    map_file_  = this->get_parameter("map_file").as_string();
    frame_id_  = this->get_parameter("frame_id").as_string();
    origin_lat_ = this->get_parameter("origin_lat").as_double();
    origin_lon_ = this->get_parameter("origin_lon").as_double();
    projParameter_ = this->get_parameter("projParameter").as_string();
    double hz  = this->get_parameter("publish_rate_hz").as_double();
    points_x_  = getParameterDoubleArray(this, "points_x");
    points_y_  = getParameterDoubleArray(this, "points_y");
    points_z_  = getParameterDoubleArray(this, "points_z");
    spheres_scale_ = this->get_parameter("spheres_scale").as_double();
    spheres_color_r_ = this->get_parameter("spheres_color_r").as_double();
    spheres_color_g_ = this->get_parameter("spheres_color_g").as_double();
    spheres_color_b_ = this->get_parameter("spheres_color_b").as_double();
    spheres_color_a_ = this->get_parameter("spheres_color_a").as_double();

    // ---- Publisher --------------------------------------------------------
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "lanelet2_map_markers", rclcpp::QoS(1).transient_local());
    spheres_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "lanelet2_map_spheres", rclcpp::QoS(1).transient_local());

    // ---- Load map ---------------------------------------------------------
    if (!loadMap()) {
      RCLCPP_ERROR(get_logger(), "Failed to load map — node will not publish.");
      return;
    }

    buildMarkers();

    // ---- Publish once immediately, then on a timer ------------------------
    publishMarkers();
    auto period = std::chrono::duration<double>(1.0 / hz);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      [this]() { publishMarkers(); });
  }

private:
  // ---- Members ------------------------------------------------------------
  std::string map_file_, frame_id_, projParameter_;
  double origin_lat_{0.0}, origin_lon_{0.0};
  std::vector<double> points_x_, points_y_, points_z_;
  double spheres_scale_{0.3};
  double spheres_color_r_{1.0}, spheres_color_g_{0.5}, spheres_color_b_{0.0}, spheres_color_a_{1.0};

  lanelet::LaneletMapPtr map_;
  visualization_msgs::msg::MarkerArray marker_array_;
  visualization_msgs::msg::MarkerArray spheres_marker_array_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr spheres_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- Load ---------------------------------------------------------------
  bool loadMap()
  {
    RCLCPP_INFO(get_logger(), "Loading Lanelet2 map: %s", map_file_.c_str());
    RCLCPP_INFO(get_logger(), "PROJ parameter: %s", projParameter_.c_str());

    try {
      lanelet::GPSPoint gps_origin{origin_lat_, origin_lon_, 0.0};
      lanelet::Origin origin(gps_origin);
      ProjProjector projector(projParameter_, origin);

      lanelet::ErrorMessages errors;
      map_ = lanelet::load(map_file_, projector, &errors);

      for (const auto & e : errors) {
        RCLCPP_WARN(get_logger(), "Map load warning: %s", e.c_str());
      }

      RCLCPP_INFO(get_logger(),
        "Map loaded — lanelets: %zu  linestrings: %zu  areas: %zu",
        map_->laneletLayer.size(),
        map_->lineStringLayer.size(),
        map_->areaLayer.size());
      return true;
    }
    catch (const std::exception & ex) {
      RCLCPP_ERROR(get_logger(), "Exception loading map: %s", ex.what());
      return false;
    }
  }

  // ---- Build markers once -------------------------------------------------
  void buildMarkers()
  {
    marker_array_.markers.clear();
    spheres_marker_array_.markers.clear();
    int id = 0;
    auto stamp = this->now();

    // ------------------------------------------------------------------
    // 1. Lanelets — left boundary (white), right boundary (yellow),
    //               centre-line (green dashed)
    // ------------------------------------------------------------------
    for (const auto & ll : map_->laneletLayer) {

      // Left boundary
      addLineStripMarker(ll.leftBound(),  id++, stamp,
        "lanelet_left",   makeColor(1,1,1,0.9f), 0.10f);

      // Right boundary
      addLineStripMarker(ll.rightBound(), id++, stamp,
        "lanelet_right",  makeColor(1,1,0,0.9f), 0.10f);

      // Centre-line (computed as average of left/right)
      addCentreLineMarker(ll, id++, stamp);

      // Lanelet ID text label at the centre
      addTextMarker(centreOf(ll), std::to_string(ll.id()),
        id++, stamp, "lanelet_ids", makeColor(0.9f,0.9f,0.9f,1.0f), 0.6f);
    }

    // ------------------------------------------------------------------
    // 2. Standalone line-strings (road-edge, stop-line, pedestrian, …)
    // ------------------------------------------------------------------
    for (const auto & ls : map_->lineStringLayer) {
      // Skip linestrings already drawn as lanelet bounds
      std::string type = ls.hasAttribute("type")
        ? ls.attribute("type").value()
        : "";

      std_msgs::msg::ColorRGBA color;
      float width = 0.06f;

      if (type == "stop_line")             color = makeColor(1,0,0,1);
      else if (type == "pedestrian_marking") color = makeColor(0,1,1,1);
      else if (type == "road_border")      color = makeColor(0.5f,0.5f,0.5f,0.8f);
      else                                 color = makeColor(0.4f,0.4f,1.0f,0.5f);

      addLineStripMarker(ls, id++, stamp, "linestrings", color, width);
    }

    // ------------------------------------------------------------------
    // 3. Areas (parking lots, intersections, …)
    // ------------------------------------------------------------------
    for (const auto & area : map_->areaLayer) {
      for (const auto & ls : area.outerBound()) {
        addLineStripMarker(ls, id++, stamp,
          "area_bounds", makeColor(1,0.5f,0,0.8f), 0.08f);
      }
    }

    // ------------------------------------------------------------------
    // 4. Points/Vertices (represented as spheres)
    // ------------------------------------------------------------------
    size_t num_points = std::min({points_x_.size(), points_y_.size(), points_z_.size()});
    for (size_t i = 0; i < num_points; ++i) {
      visualization_msgs::msg::Marker m = baseMarker(static_cast<int>(i), stamp, "lanelet_points",
        visualization_msgs::msg::Marker::SPHERE);
      m.pose.position.x = points_x_[i];
      m.pose.position.y = points_y_[i];
      m.pose.position.z = points_z_[i];
      m.scale.x = spheres_scale_;
      m.scale.y = spheres_scale_;
      m.scale.z = spheres_scale_;
      m.color = makeColor(spheres_color_r_, spheres_color_g_, spheres_color_b_, spheres_color_a_);
      spheres_marker_array_.markers.push_back(std::move(m));
    }

    RCLCPP_INFO(get_logger(), "Built %zu markers and %zu spheres.",
      marker_array_.markers.size(), spheres_marker_array_.markers.size());
  }

  // ---- Publish ------------------------------------------------------------
  void publishMarkers()
  {
    // Refresh header stamps so RViz doesn't complain about old data
    auto now = this->now();
    for (auto & m : marker_array_.markers) {
      m.header.stamp = now;
    }
    marker_pub_->publish(marker_array_);

    for (auto & m : spheres_marker_array_.markers) {
      m.header.stamp = now;
    }
    spheres_pub_->publish(spheres_marker_array_);
  }

  // =========================================================================
  // Marker helpers
  // =========================================================================

  // Base marker with common fields filled in
  visualization_msgs::msg::Marker baseMarker(
    int id,
    const rclcpp::Time & stamp,
    const std::string & ns,
    int32_t type)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = frame_id_;
    m.header.stamp    = stamp;
    m.ns              = ns;
    m.id              = id;
    m.type            = type;
    m.action          = visualization_msgs::msg::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.lifetime        = rclcpp::Duration(0, 0);   // forever
    return m;
  }

  // LINE_STRIP from any linestring-like container
  template<typename LineStringT>
  void addLineStripMarker(
    const LineStringT & ls,
    int id,
    const rclcpp::Time & stamp,
    const std::string & ns,
    const std_msgs::msg::ColorRGBA & color,
    float width)
  {
    if (ls.size() < 2) return;

    auto m = baseMarker(id, stamp, ns,
      visualization_msgs::msg::Marker::LINE_STRIP);
    m.scale.x = width;
    m.color   = color;

    for (const auto & pt : ls) {
      m.points.push_back(toPoint(pt));
    }
    marker_array_.markers.push_back(std::move(m));
  }

  // Dashed centre-line (alternating point pairs)
  void addCentreLineMarker(
    const lanelet::ConstLanelet & ll,
    int id,
    const rclcpp::Time & stamp)
  {
    const auto & left  = ll.leftBound();
    const auto & right = ll.rightBound();
    size_t n = std::min(left.size(), right.size());
    if (n < 2) return;

    auto m = baseMarker(id, stamp, "lanelet_centre",
      visualization_msgs::msg::Marker::LINE_LIST);
    m.scale.x = 0.05f;
    m.color   = makeColor(0,1,0,0.7f);

    // Interpolate midpoints and draw dashes
    std::vector<geometry_msgs::msg::Point> pts;
    pts.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      geometry_msgs::msg::Point p;
      p.x = (left[i].x() + right[i].x()) * 0.5;
      p.y = (left[i].y() + right[i].y()) * 0.5;
      p.z = (left[i].z() + right[i].z()) * 0.5;
      pts.push_back(p);
    }
    // Emit every other segment as a dash
    for (size_t i = 0; i + 1 < pts.size(); i += 2) {
      m.points.push_back(pts[i]);
      m.points.push_back(pts[i+1]);
    }
    if (!m.points.empty())
      marker_array_.markers.push_back(std::move(m));
  }

  // Text label
  void addTextMarker(
    const geometry_msgs::msg::Point & pos,
    const std::string & text,
    int id,
    const rclcpp::Time & stamp,
    const std::string & ns,
    const std_msgs::msg::ColorRGBA & color,
    float size)
  {
    auto m = baseMarker(id, stamp, ns,
      visualization_msgs::msg::Marker::TEXT_VIEW_FACING);
    m.pose.position = pos;
    m.scale.z       = size;
    m.color         = color;
    m.text          = text;
    marker_array_.markers.push_back(std::move(m));
  }

  // Geometric centre of a lanelet
  geometry_msgs::msg::Point centreOf(const lanelet::ConstLanelet & ll)
  {
    const auto & left  = ll.leftBound();
    const auto & right = ll.rightBound();
    size_t n = std::min(left.size(), right.size());
    size_t mid = n / 2;

    geometry_msgs::msg::Point p;
    if (n == 0) return p;
    p.x = (left[mid].x() + right[mid].x()) * 0.5;
    p.y = (left[mid].y() + right[mid].y()) * 0.5;
    p.z = (left[mid].z() + right[mid].z()) * 0.5 + 0.3;  // float slightly above
    return p;
  }
};

// ---------------------------------------------------------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Lanelet2Visualizer>());
  rclcpp::shutdown();
  return 0;
}