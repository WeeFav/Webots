from proto_nodes import extract_lanes

import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import shutil
from pathlib import Path
import re
import argparse

POSE_AXIS = np.array([0.707107, -0.707107, 0.0])
POSE_ANGLE = 3.14157
POSE_ROT = R.from_rotvec(POSE_AXIS * POSE_ANGLE)

def inverse_transform(points, translation, rotation, extra_rot=None):
    """
    Undo Webots transform:
    local = inverse(rot) * (world - translation)
    """

    axis = rotation[:3]
    angle = rotation[3]

    rot = R.from_rotvec(np.array(axis) * angle)

    if extra_rot is not None:
        rot = rot * extra_rot

    rot_inv = rot.inv()

    result = []

    for p in points:
        p_local = rot_inv.apply(np.array(p) - np.array(translation))
        result.append(p_local)

    return np.array(result)

# ----------------------------
# Recompute SAME bounds as SVG export
# ----------------------------
def apply_transform(points, translation, rotation, extra_rot=None):
    """
    Apply Webots translation + axis-angle rotation.
    """
    axis = rotation[:3]
    angle = rotation[3]

    rot = R.from_rotvec(np.array(axis) * angle)
    if extra_rot is not None:
        rot = rot * extra_rot   # IMPORTANT: Webots order
        
    transformed = []

    for p in points:
        p_rot = rot.apply(p)
        p_world = p_rot + np.array(translation)
        transformed.append(p_world)

    return np.array(transformed)


def compute_bounds(roads, crossroads):
    """
    Compute world bounds for SVG sizing.
    """

    all_pts = []

    for road in roads:
        pts = apply_transform(
            road.wayPoints,
            road.translation,
            road.rotation
        )
        if pts.size > 0:
            all_pts.extend(pts[:, :2])

    for cross in crossroads:
        pts = apply_transform(
            cross.shape,
            cross.translation,
            cross.rotation
        )
        if pts.size > 0:
            all_pts.extend(pts[:, :2])

    all_pts = np.array(all_pts)

    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    return min_xy, max_xy

# ----------------------------
# SVG → World coordinate transform
# ----------------------------
def svg_to_world(p, min_xy, max_xy, scale, margin, height):
    x_svg, y_svg = p

    x_world = (x_svg - margin) / scale + min_xy[0]

    # invert y-axis
    y_world = ((height - y_svg - margin) / scale) + min_xy[1]

    return [x_world, y_world, 0.0]

# ----------------------------
# Parse SVG path string
# ----------------------------
def parse_path(d):
    """
    Supports:
        M/m
        L/l
        Z/z
        implicit line commands

    Returns absolute SVG coordinates.
    """

    tokens = re.findall(r'[MmLlZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?', d)

    points = []

    i = 0

    current = np.array([0.0, 0.0])

    command = None

    while i < len(tokens):

        token = tokens[i]

        # command token
        if re.match(r'[MmLlZz]', token):
            command = token
            i += 1

            if command in ['Z', 'z']:
                break

            continue

        # coordinate pair
        x = float(tokens[i])
        y = float(tokens[i + 1])

        pt = np.array([x, y])

        # absolute
        if command in ['M', 'L']:
            current = pt

        # relative
        elif command in ['m', 'l']:
            current = current + pt

        points.append(tuple(current))

        i += 2

        # SVG rule:
        # after initial M/m,
        # remaining pairs are treated as L/l
        if command == 'M':
            command = 'L'
        elif command == 'm':
            command = 'l'

    return points

# ----------------------------
# Main conversion
# ----------------------------
def svg_to_webots(svg_file, wbt_file, scale=5.0, margin=50):
    roads, crossroads = extract_lanes(wbt_file)

    # compute same bounds used in export
    min_xy, max_xy = compute_bounds(roads, crossroads)

    # IMPORTANT: SVG canvas height must match exporter
    height = (max_xy[1] - min_xy[1]) * scale + margin * 2

    tree = ET.parse(svg_file)
    root = tree.getroot()

    # namespace-safe tag check
    def is_path(elem):
        return elem.tag.endswith("path")
    
    road_updates = {}

    for elem in root.iter():
        if not is_path(elem):
            continue

        path_id = elem.attrib.get("id", None)
        if path_id is None:
            continue

        d = elem.attrib.get("d", "")
        svg_points = parse_path(d)

        # ----------------------------
        # ROAD UPDATE
        # ----------------------------
        road = next((r for r in roads if str(r.id) == str(path_id)), None)
        if road is not None:
            world_pts = [
                svg_to_world(p, min_xy, max_xy, scale, margin, height)
                for p in svg_points
            ]

            local_pts = inverse_transform(
                world_pts,
                road.translation,
                road.rotation
            )

            road.wayPoints = local_pts.tolist()
            road_updates[str(road.id)] = (road.wayPoints, 'r')
            continue

        # ----------------------------
        # CROSSROAD UPDATE
        # ----------------------------
        cross = next((c for c in crossroads if str(c.id) == str(path_id)), None)
        if cross is not None:
            world_pts = [
                svg_to_world(p, min_xy, max_xy, scale, margin, height)
                for p in svg_points
            ]

            local_pts = inverse_transform(
                world_pts,
                cross.translation,
                cross.rotation,
                extra_rot=POSE_ROT
            )

            cross.shape = local_pts.tolist()
            road_updates[str(cross.id)] = (cross.shape, 'c')
            continue

        print(f"[WARN] No match for SVG id: {path_id}")

    print("SVG → Webots conversion complete.")
    return road_updates

def format_points(points):
    """
    Convert list of points into Webots format.
    Supports:
        (x, y)
        (x, y, z)
    """

    lines = []

    for p in points:
        if len(p) == 2:
            x, y = p
            z = 0.0
        else:
            x, y, z = p

        lines.append(f"    {x:.6f} {y:.6f} {z:.6f}")

    return "\n".join(lines)

def replace_field_block(node_block, field_name, new_points):
    """
    Replace contents inside:
        field_name [
            ...
        ]
    """

    field_start = node_block.find(f"{field_name} [")

    if field_start == -1:
        return node_block

    content_start = node_block.find("[", field_start) + 1
    content_end = node_block.find("]", content_start)

    new_content = "\n" + format_points(new_points) + "\n  "

    return (
        node_block[:content_start]
        + new_content
        + node_block[content_end:]
    )


def write_wbt(road_updates, wbt_file, output_file):
    """
    road_updates format:

    {
        road_id: (points, type)

        type:
            'r' -> Road      -> update wayPoints
            'c' -> Crossroad -> update shape
    }
    """

    shutil.copy(wbt_file, output_file)

    text = Path(output_file).read_text(encoding="utf-8")

    pos = 0
    output = []

    while True:

        # find next Road or Crossroad
        road_pos = text.find("Road {", pos)
        cross_pos = text.find("Crossroad {", pos)

        candidates = [p for p in [road_pos, cross_pos] if p != -1]

        if not candidates:
            output.append(text[pos:])
            break

        node_start = min(candidates)

        output.append(text[pos:node_start])

        # determine node type
        if node_start == road_pos:
            node_keyword = "Road {"
        else:
            node_keyword = "Crossroad {"

        # find matching brace
        brace_count = 0
        node_end = None

        for i in range(node_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1

                if brace_count == 0:
                    node_end = i
                    break

        if node_end is None:
            raise ValueError("Unmatched braces in WBT")

        node_block = text[node_start:node_end + 1]

        # extract id
        id_key = 'id "'
        id_pos = node_block.find(id_key)

        if id_pos != -1:

            id_start = id_pos + len(id_key)
            id_end = node_block.find('"', id_start)

            node_id = node_block[id_start:id_end]

            if node_id in road_updates:

                points, geom_type = road_updates[node_id]

                # road -> update wayPoints
                if geom_type == 'r':

                    node_block = replace_field_block(
                        node_block,
                        "wayPoints",
                        points
                    )

                    print(f"Updated Road {node_id}")

                # crossroad -> update shape
                elif geom_type == 'c':

                    node_block = replace_field_block(
                        node_block,
                        "shape",
                        points
                    )

                    print(f"Updated Crossroad {node_id}")

        output.append(node_block)

        pos = node_end + 1

    updated_text = "".join(output)

    Path(output_file).write_text(updated_text, encoding="utf-8")

    print("Finished updating WBT")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg")
    parser.add_argument("--wbt")
    parser.add_argument("--output")
    args = parser.parse_args()
        
    road_updates = svg_to_webots(args.svg, args.wbt)
    write_wbt(road_updates, args.wbt, args.output)