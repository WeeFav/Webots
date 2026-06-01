import numpy as np
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import shutil
from pathlib import Path
import re
import argparse

from webots_to_svg import extract_proto, compute_bounds

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
    roads, crossroads, forests, buildings, parkings, waters = extract_proto(wbt_file)

    # compute same bounds used in export
    min_xy, max_xy = compute_bounds(roads, crossroads, forests, buildings, parkings, waters)

    # IMPORTANT: SVG canvas height must match exporter
    height = (max_xy[1] - min_xy[1]) * scale + margin * 2

    tree = ET.parse(svg_file)
    root = tree.getroot()

    # namespace-safe tag check
    def is_path(elem):
        return elem.tag.endswith("path")
    
    node_updates = {}

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
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, road.translation, road.rotation)
            road.wayPoints = local_pts.tolist()
            node_updates[str(road.id)] = (road.wayPoints, 'r')
            continue

        # ----------------------------
        # CROSSROAD UPDATE
        # ----------------------------
        cross = next((c for c in crossroads if str(c.id) == str(path_id)), None)
        if cross is not None:
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, cross.translation, cross.rotation, extra_rot=POSE_ROT)
            cross.shape = local_pts.tolist()
            node_updates[str(cross.id)] = (cross.shape, 'c')
            continue
        
        # ----------------------------
        # FOREST UPDATE
        # ----------------------------
        forest = next((f for f in forests if str(f.id) == str(path_id)), None)
        if forest is not None:
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, forest.translation, forest.rotation)
            forest.shape = [[p[0], -p[1]] for p in local_pts] # Undo the Y flip that was applied during export
            node_updates[str(forest.id)] = (forest.shape, 'f')
            continue
        
        # ----------------------------
        # BUILDING UPDATE
        # ----------------------------
        building = next((f for f in buildings if str(f.id) == str(path_id)), None)
        if building is not None:
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, building.translation, building.rotation)
            building.corners = [[p[0], p[1]] for p in local_pts]
            node_updates[str(building.id)] = (building.corners, 'b')
            continue

        # ----------------------------
        # PARKING UPDATE
        # ----------------------------
        parking = next((c for c in parkings if str(c.id) == str(path_id)), None)
        if parking is not None:
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, parking.translation, parking.rotation)
            parking.point = local_pts.tolist()
            node_updates[str(parking.id)] = (parking.point, 't')
            continue
        
        # ----------------------------
        # WATER UPDATE
        # ----------------------------
        water = next((c for c in waters if str(c.id) == str(path_id)), None)
        if water is not None:
            world_pts = [svg_to_world(p, min_xy, max_xy, scale, margin, height) for p in svg_points]
            local_pts = inverse_transform(world_pts, water.translation, water.rotation)
            water.point = local_pts.tolist()
            node_updates[str(water.id)] = (water.point, 't')
            continue

        print(f"[WARN] No match for SVG id: {path_id}")

    print("SVG → Webots conversion complete.")
    return node_updates

def format_3d_points(points):
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

def format_2d_points(points):
    """
    Convert points into 2d format:

    shape [
      x y,
      x y,
      ...
    ]
    """

    entries = []

    for p in points:
        x, y = p[:2]
        entries.append(f"    {x:.6f} {y:.6f},")

    return "\n".join(entries)

def replace_field_block(node_block, field_name, new_points, type):
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

    if type == 'f' or type == 'b':
        new_content = "\n" + format_2d_points(new_points) + "\n  "
    else:
        new_content = "\n" + format_3d_points(new_points) + "\n  "

    return (node_block[:content_start] + new_content + node_block[content_end:])

def write_wbt(node_updates, wbt_file, output_file):
    shutil.copy(wbt_file, output_file)
    text = Path(output_file).read_text(encoding="utf-8")

    pos = 0
    output = []

    while True:
        pattern = re.compile(r"^\s*(Road|Crossroad|Forest|SimpleBuilding|Transform)\s*\{", re.MULTILINE)        
        
        match = pattern.search(text, pos)
        if not match:
            output.append(text[pos:])
            break

        node_start = match.start()

        output.append(text[pos:node_start])

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

            if node_id in node_updates:
                points, geom_type = node_updates[node_id]

                if geom_type == 'r':
                    node_block = replace_field_block(node_block, "wayPoints", points, geom_type)
                    print(f"Updated Road {node_id}")
                elif geom_type == 'c':
                    node_block = replace_field_block(node_block, "shape", points, geom_type)
                    print(f"Updated Crossroad {node_id}")
                elif geom_type == 'f':
                    node_block = replace_field_block(node_block, "shape", points, geom_type)
                    print(f"Updated Forest {node_id}")
                elif geom_type == 'b':
                    node_block = replace_field_block(node_block, "corners", points, geom_type)
                    print(f"Updated Building {node_id}")
                elif geom_type == 't':
                    node_block = replace_field_block(node_block, "point", points, geom_type)
                    print(f"Updated Transform {node_id}")

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
        
    node_updates = svg_to_webots(args.svg, args.wbt)
    write_wbt(node_updates, args.wbt, args.output)