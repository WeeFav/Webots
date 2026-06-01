import os
import re
import matplotlib.pyplot as plt
import numpy as np
from proto_nodes import RoadPROTO, CrossroadPROTO, ForestPROTO, SimpleBuildingPROTO

def parse_vector_block(block):
    """
    Parse blocks containing vectors.

    Examples accepted:

    shape [
      -41.23 -170.35,
      57.90 41.05,
      55.73 41.97,
    ]

    wayPoints [
      0.0 0.0 0.0
      2.7 -4.0 0.0
      61.3 -122.9 0.0
    ]

    Returns:
        [
            [-41.23, -170.35],
            [57.90, 41.05],
            ...
        ]

    or

        [
            [0.0, 0.0, 0.0],
            [2.7, -4.0, 0.0],
            ...
        ]
    """

    # normalize commas to newlines
    block = block.replace(",", "\n")

    vectors = []

    for line in block.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            vector = [float(x) for x in line.split()]
            vectors.append(vector)
        except ValueError:
            pass

    return vectors

def parse_string_list(block):
    """
    Parse bracket blocks containing quoted strings.

    Example:
        "1"
        "2"

    Returns:
        ["1", "2"]
    """

    values = []

    for line in block.split("\n"):
        line = line.strip()

        if not line:
            continue

        if line.startswith('"') and line.endswith('"'):
            values.append(line.strip('"'))

    return values

def parse_scalar_list(block):
    """
    Parse a flat bracket block that contains only space/newline-separated
    scalars (used for roadsWidth, numberOfLanes, numberOfForwardLanes, etc.).
 
    Example block content: '7 7 7 7'  or  '2\\n2\\n2\\n2'
    Returns a list of ints when all values are whole numbers, else floats.
    """
    values = []
    for token in block.split():
        try:
            f = float(token)
            values.append(int(f) if f == int(f) else f)
        except ValueError:
            pass
    return values

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

def parse_block(block):
    """
    Convert webots fields
    translation x y z
    id "123"
    shape [
        x y z
        x y z
    ]
    into python dict
    result = {
        "translation": [x, y, z],
        "id": "123,
        shape: [[x,y,z], [x,y,z]]
    }
    
    Currently supported fields: 
        - wayPoints []
        - shape []
        - single line fields (SFBool, SFString, SFFloat, SFInt)
    """
    result = {}

    # -------------------------------------------------
    # wayPoints [ ... ]
    # -------------------------------------------------
    waypoints_match = re.search(r"wayPoints\s*\[(.*?)\]", block, re.DOTALL)
    if waypoints_match:
        result["wayPoints"] = parse_vector_block(waypoints_match.group(1))
        block = (block[:waypoints_match.start()] + block[waypoints_match.end():]) # remove "wayPoints []" from text

    # -------------------------------------------------
    # shape [ ... ]
    # -------------------------------------------------
    shape_match = re.search(r"shape\s*\[(.*?)\]", block, re.DOTALL)
    if shape_match:
        result["shape"] = parse_vector_block(shape_match.group(1))
        block = (block[:shape_match.start()] + block[shape_match.end():]) # remove "shape []" from text
    
    # -------------------------------------------------
    # corners [ ... ]
    # -------------------------------------------------
    corners_match = re.search(r"corners\s*\[(.*?)\]", block, re.DOTALL)
    if corners_match:
        result["corners"] = parse_vector_block(corners_match.group(1))
        block = (block[:corners_match.start()] + block[corners_match.end():]) # remove "shape []" from text
    
    # -------------------------------------------------
    # appearance SomeNodeType { ... }
    # -------------------------------------------------
    appearance_match = re.search(r"appearance\s+(\w+)\s*\{.*?\}", block, re.DOTALL)
    if appearance_match:
        result["appearance"] = appearance_match.group(1)
        block = (block[:appearance_match.start()] + block[appearance_match.end():])
    
    # -------------------------------------------------
    # remaining single-line fields (e.g. "width 7", "translation 1 2 3")
    # -------------------------------------------------
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        parts = line.split(None, 1)
        if len(parts) != 2:
            continue

        key, val = parts # ["width","0.15"]
        result[key] = parse_value(val)

    return result

def extract_blocks(text, node_name):
    """
    Get everything inside node_name
    Road {
        block = ...
    }
    blocks = [block, block, ...]
    """
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
    
def extract_proto(wbt_path):    
    with open(wbt_path, "r", encoding="utf-8") as f:
        text = f.read()

    results = [] # list of dict

    for node_name in ["Road", "Crossroad", "Forest", "SimpleBuilding"]:
        blocks = extract_blocks(text, node_name)

        for b in blocks:
            parsed = parse_block(b)
            parsed["type"] = node_name  # tag the type
            results.append(parsed)
    
    roads = []
    crossroads = []
    forests = []
    buildings = []
    
    for data in results: 
        if data["type"] == 'Road':
            road = RoadPROTO(data) 
            roads.append(road)
        elif data["type"] == 'Crossroad':
            crossroad = CrossroadPROTO(data)
            crossroads.append(crossroad)
        elif data["type"] == 'Forest':
            forest = ForestPROTO(data)
            forests.append(forest)
        elif data["type"] == 'SimpleBuilding':
            building = SimpleBuildingPROTO(data)
            buildings.append(building)
        else:
            continue

    return roads, crossroads, forests, buildings