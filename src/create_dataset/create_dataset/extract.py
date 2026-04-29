import os
import re
import matplotlib.pyplot as plt
from create_dataset.proto_nodes import StraightRoadSegmentPROTO, RoadPROTO, RoadLine, CurvedRoadSegmentPROTO
import numpy as np

def extract_blocks(text, node_name):
    blocks = []
    pattern = re.compile(rf"{node_name}\s*\{{")

    for match in pattern.finditer(text):
        start = match.end()
        brace_count = 1
        i = start

        while i < len(text) and brace_count > 0:
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
            i += 1

        block = text[start:i-1].strip()
        blocks.append(block)

    return blocks


def parse_value(value):
    value = value.strip()

    # string
    if value.startswith('"') and value.endswith('"'):
        return value.strip('"')

    # boolean
    if value in ["TRUE", "FALSE"]:
        return value == "TRUE"

    # vector (e.g. translation)
    if re.match(r"^-?\d+(\.\d+)?(\s+-?\d+(\.\d+)?)+$", value):
        return [float(x) for x in value.split()]

    # number
    try:
        if "." in value:
            return float(value)
        return int(value)
    except:
        return value


def parse_lines_block(block):
    lines = []
    pattern = re.compile(r"RoadLine\s*\{([^}]*)\}", re.DOTALL)

    for match in pattern.finditer(block):
        content = match.group(1)
        line_dict = {}

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                key, val = parts
                line_dict[key] = parse_value(val)

        # Map dict → RoadLine object with defaults
        road_line = RoadLine(
            color=line_dict.get("color", (1, 1, 1)),
            type=line_dict.get("type", "dashed"),
            width=line_dict.get("width", 0.15)
        )

        lines.append(road_line)

    return lines


def parse_block(block):
    result = {}
    lines_match = re.search(r"lines\s*\[(.*?)\]", block, re.DOTALL)

    # Handle lines separately
    if lines_match:
        result["lines"] = parse_lines_block(lines_match.group(1))
        block = block[:lines_match.start()] + block[lines_match.end():]

    # Parse remaining fields
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue

        key, val = parts
        result[key] = parse_value(val)

    return result


def extract_all_road_segments(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    result = []

    for node_name in ["StraightRoadSegment", "CurvedRoadSegment"]:
        blocks = extract_blocks(text, node_name)

        for b in blocks:
            parsed = parse_block(b)
            parsed["type"] = node_name  # tag the type
            result.append(parsed)

    return result

def plot_lanes(lanes):
    for j in range(len(lanes)):  # for each lane
        center_x = []
        center_y = []

        for i in range(len(lanes[j])):
            (x, y, _) = lanes[j][i]
            center_x.append(x)
            center_y.append(y)

        # plot lane centerline
        plt.plot(center_x, center_y, marker='o')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Lane Centerlines")
    plt.axis("equal")  # keep geometry correct
    plt.show()
    
def interpolate_lane_3d(points, num_points=100):
    """
    points: list of (x, y, z) or (N, 3)
    returns: (num_points, 3)
    """

    points = np.asarray(points, dtype=np.float32)

    if len(points) < 2:
        return points

    # Compute segment distances in 3D
    deltas = np.diff(points, axis=0)
    dists = np.linalg.norm(deltas, axis=1)

    # Cumulative distance
    cumdist = np.insert(np.cumsum(dists), 0, 0)

    # Normalize to [0, 1]
    t = cumdist / cumdist[-1]

    # New evenly spaced samples
    t_new = np.linspace(0, 1, num_points)

    # Interpolate x, y, z
    x_new = np.interp(t_new, t, points[:, 0])
    y_new = np.interp(t_new, t, points[:, 1])
    z_new = np.interp(t_new, t, points[:, 2])

    return np.stack([x_new, y_new, z_new], axis=1)
    
def extract_lanes(wbt_path):
    d = {
        '0': 'y',
        '1': 'y',
        '2': 'y',
        '3': 'o',
        '4': 'o',
        '5': 'r', 
        '6': 'r', 
        '7': 'r', 
        '8': 'b', 
        '9': 'b', 
        '10': 'b', 
        '11': 'g', 
        '12': 'g', 
        '13': 'g', 
        '14': 'p', 
        '15': 'p', 
    }
    
    parsed = extract_all_road_segments(wbt_path)
    
    all_lanes = []
    for data in parsed: 
        print(data)
        if data["type"] == 'StraightRoadSegment':
            straight_road = StraightRoadSegmentPROTO(data)
            lanes = straight_road.road.compute_lane_lines()
        elif data["type"] == 'CurvedRoadSegment':       
            curved_road = CurvedRoadSegmentPROTO(data)
            lanes = curved_road.road.compute_lane_lines()
            
        all_lanes.extend(lanes)
        
    all_lanes_center = []

    for lane in all_lanes:
        center = []
        for point_l, point_r in lane:
            (x_l, y_l, z_l) = point_l  
            (x_r, y_r, z_r) = point_r

            # midpoint
            x_c = (x_l + x_r) / 2
            y_c = (y_l + y_r) / 2
            z_c = (z_l + z_r) / 2

            center.append((x_c, y_c, z_c))

        # ✅ interpolate in 3D BEFORE projection
        center_interp = interpolate_lane_3d(center, num_points=100)

        all_lanes_center.append(center_interp)   
             
    return all_lanes_center   
    # plot_lanes(all_lanes)

if __name__ == '__main__':
    all_lanes_center = extract_lanes("/home/marvin/Webots/src/create_dataset/worlds/my_world.wbt")
    plot_lanes(all_lanes_center)