import os
import re
import matplotlib.pyplot as plt
from proto_nodes import StraightRoadSegmentPROTO, RoadPROTO, RoadLine

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


def extract_straight_road_segments(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    blocks = extract_blocks(text, "StraightRoadSegment")
    return [parse_block(b) for b in blocks]

def plot_lanes(lanes):
    for j in range(len(lanes)):  # for each lane
        center_x = []
        center_y = []

        for i in range(len(lanes[j])):
            (x_l, y_l, _), (x_r, y_r, _) = lanes[j][i]

            # compute midpoint
            x_c = (x_l + x_r) / 2
            y_c = (y_l + y_r) / 2

            center_x.append(x_c)
            center_y.append(y_c)

        # plot lane centerline
        plt.plot(center_x, center_y, marker='o')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Lane Centerlines")
    plt.axis("equal")  # keep geometry correct
    plt.show()

if __name__ == '__main__':
    wbt_path = "simple.wbt"
    datas = extract_straight_road_segments(wbt_path)
    
    all_lanes = []
    for data in datas:        
        straight_road = StraightRoadSegmentPROTO(data)
        lanes = straight_road.road.compute_lane_lines()
        all_lanes.extend(lanes)
        
    plot_lanes(all_lanes)
