#include <rclcpp/rclcpp.hpp>
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

#include <chrono>
#include <string>
#include <memory>

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
    this->declare_parameter<double>("publish_rate_hz",  1.0);

    map_file_  = this->get_parameter("map_file").as_string();
    frame_id_  = this->get_parameter("frame_id").as_string();
    origin_lat_ = this->get_parameter("origin_lat").as_double();
    origin_lon_ = this->get_parameter("origin_lon").as_double();
    double hz  = this->get_parameter("publish_rate_hz").as_double();

    // ---- Publisher --------------------------------------------------------
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
      "lanelet2_map_markers", rclcpp::QoS(1).transient_local());

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
  std::string map_file_, frame_id_;
  double origin_lat_{0.0}, origin_lon_{0.0};

  lanelet::LaneletMapPtr map_;
  visualization_msgs::msg::MarkerArray marker_array_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---- Load ---------------------------------------------------------------
  bool loadMap()
  {
    RCLCPP_INFO(get_logger(), "Loading Lanelet2 map: %s", map_file_.c_str());
    RCLCPP_INFO(get_logger(), "Origin  lat=%.6f  lon=%.6f", origin_lat_, origin_lon_);

    try {
      // UTM projector anchored at the supplied origin
      lanelet::projection::UtmProjector projector(
        lanelet::Origin({origin_lat_, origin_lon_}));

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

    RCLCPP_INFO(get_logger(), "Built %zu markers.", marker_array_.markers.size());
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