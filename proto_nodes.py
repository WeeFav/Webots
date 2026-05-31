import numpy as np
import math
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as R

def b_spline_3(points, subdivision):
    """
    Approximation of Webots wbgeometry.bSpline3 using SciPy.
    
    points: list of (x,y,z)
    subdivision: number of points per segment
    """

    pts = np.array(points).T
    x, y, z = pts

    # k=3 → cubic B-spline (same as Webots behavior)
    tck, u = splprep([x, y, z], s=0, k=3)

    # number of output samples
    num_samples = len(points) * subdivision

    u_new = np.linspace(0, 1, num_samples)

    x_new, y_new, z_new = splev(u_new, tck)

    return [
        (float(x_new[i]), float(y_new[i]), float(z_new[i]))
        for i in range(len(x_new))
    ]

def vec2_angle(a, b):
    """Equivalent to wbvector2.angle(a, b)"""
    return math.atan2(a[1] - b[1], a[0] - b[0])

def vec3_distance(a, b):
    """Euclidean distance (wbvector3.distance equivalent)."""
    return math.sqrt(
        (a[0] - b[0])**2 +
        (a[1] - b[1])**2 +
        (a[2] - b[2])**2
    )

class RoadLine:
    def __init__(self, color=(1, 1, 1), type="dashed", width=0.15):
        self.color = color
        self.type = type
        self.width = width

class RoadPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.name = "road"
        self.id = ""
        self.startJunction = ""
        self.endJunction = ""

        self.width = 7.0
        self.numberOfLanes = 2
        self.numberOfForwardLanes = 1
        self.speedLimit = -1.0

        self.lines = [RoadLine()]

        self.roadBorderHeight = 0.15
        self.roadBorderWidth = [0.8]

        self.road = True
        self.rightBorder = True
        self.leftBorder = True
        self.rightBarrier = False
        self.leftBarrier = False
        self.bottom = False

        self.wayPoints = [
            [0, 0, 0],
            [1, 0, 0]
        ]

        self.roadTilt = [0, 0]
        self.startingAngle = []
        self.endingAngle = []

        self.startLine = []
        self.endLine = []

        self.splineSubdivision = 4

        self.appearance = "Asphalt"
        self.pavementAppearance = "Pavement"

        self.bottomTexture = []

        self.turnLanesForward = ""
        self.turnLanesBackward = ""

        self.locked = True

        self.roadBoundingObject = False
        self.rightBorderBoundingObject = False
        self.leftBorderBoundingObject = False
        self.rightBarrierBoundingObject = True
        self.leftBarrierBoundingObject = True

        self.castShadows = False

        self.contactMaterial = "default"

        # --- override with input dict ---
        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                if key in ["startingAngle", "endingAngle"]:
                    setattr(self, key, [value])
                else:
                    setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    
    def preprocess_road(self):
        defaultLineWidth = 0.075
        heightOffset = 0.01
        wayPoints = self.wayPoints
        nbWayPoint = len(wayPoints)

        # Tilt
        originalTilt = self.roadTilt
        for j in range(nbWayPoint):
            if j >= len(originalTilt) or originalTilt[j] is None:
                if j < len(originalTilt):
                    originalTilt[j] = 0
                else:
                    originalTilt.append(0)

        originalTilt.append(originalTilt[-1])

        splineSubdivision = self.splineSubdivision

        if splineSubdivision > 0:
            wayPoints = b_spline_3(wayPoints, splineSubdivision)
            nbWayPoint = len(wayPoints)

        # Tilt interpolation
        tilts = [0] * nbWayPoint

        if splineSubdivision > 1:
            for j in range(nbWayPoint):
                ratio = (j % splineSubdivision) / splineSubdivision
                index = j // splineSubdivision

                t0 = originalTilt[index]
                t1 = originalTilt[min(index + 1, len(originalTilt) - 1)]

                tilts[j] = t0 * (1 - ratio) + t1 * ratio
        else:
            tilts = originalTilt[:nbWayPoint]

        # Angle computation (IMPORTANT PART)
        angles = [0] * nbWayPoint
        distances = [0] * nbWayPoint

        startingAngle = self.startingAngle
        endingAngle = self.endingAngle

        for i in range(nbWayPoint):
            # CASE 1: END REGION
            if (
                (i == nbWayPoint - 1 or i >= (nbWayPoint - 1 - splineSubdivision))
                and endingAngle and len(endingAngle) > 0
            ):
                ratio = 0.0
                if splineSubdivision > 0:
                    ratio = ((nbWayPoint - 1 - i) / splineSubdivision) ** 3

                # direction reference
                if i == 0:
                    ref_angle = vec2_angle(wayPoints[i + 1], wayPoints[i])
                else:
                    ref_angle = vec2_angle(wayPoints[i], wayPoints[i - 1])

                angles[i] = endingAngle[0] * (1 - ratio) - ref_angle * ratio

            # CASE 2: MIDDLE REGION
            elif i > 0 and i > splineSubdivision:
                if i == nbWayPoint - 1:
                    angles[i] = -vec2_angle(
                        wayPoints[i],
                        wayPoints[i - 1]
                    )
                else:
                    angles[i] = -vec2_angle(
                        wayPoints[i + 1],
                        wayPoints[i - 1]
                    )

            # CASE 3: START REGION
            elif startingAngle and len(startingAngle) > 0:
                ratio = 0.0
                if splineSubdivision > 0:
                    ratio = (i / splineSubdivision) ** 3

                if i == nbWayPoint - 1:
                    ref_angle = vec2_angle(wayPoints[i], wayPoints[i - 1])
                else:
                    ref_angle = vec2_angle(wayPoints[i + 1], wayPoints[i])

                angles[i] = startingAngle[0] * (1 - ratio) - ref_angle * ratio

            # CASE 4: DEFAULT (no boundary constraints)
            else:
                if i == nbWayPoint - 1:
                    angles[i] = -vec2_angle(
                        wayPoints[i],
                        wayPoints[i - 1]
                    )
                else:
                    angles[i] = -vec2_angle(
                        wayPoints[i + 1],
                        wayPoints[i]
                    )

            # Distance accumulation
            if i > 0:
                distances[i] = distances[i - 1] + vec3_distance(
                    wayPoints[i],
                    wayPoints[i - 1]
                )

        # Output core structures
        return defaultLineWidth, heightOffset, wayPoints, angles, tilts, distances

    def compute_lane_lines(self):
        """
        Returns lane boundary coordinates for each lane line.
        Output format:
        lanes[j][i] = [(x_left, y_left, z_left), (x_right, y_right, z_right)]
        """

        defaultLineWidth, heightOffset, wayPoints, angles, tilts, distances = self.preprocess_road()
        
        lanes = []

        nbWayPoint = len(wayPoints)

        for j in range(self.numberOfLanes - 1):
            # Skip invalid / empty lines
            line = self.lines[j] if j < len(self.lines) else None

            if line is not None:
                lineWidth = line.width * 0.5
                if line.type == "double":
                    lineWidth *= 3
            else:
                lineWidth = defaultLineWidth

            lane_coords = []

            # Compute lane offset (key formula)
            offset = self.width * ((j + 1) / self.numberOfLanes - 0.5)

            for i in range(nbWayPoint):
                wp = wayPoints[i]
                angle = angles[i]
                tilt = tilts[i]

                sin_a = math.sin(angle)
                cos_a = math.cos(angle)
                sin_t = math.sin(tilt)
    
                # LEFT edge of lane line
                x1 = wp[0] + sin_a * (offset - lineWidth)
                y1 = wp[1] + cos_a * (offset - lineWidth)
                z1 = wp[2] + sin_t * (offset - lineWidth) + heightOffset

                # RIGHT edge of lane line
                x2 = wp[0] + sin_a * (offset + lineWidth)
                y2 = wp[1] + cos_a * (offset + lineWidth)
                z2 = wp[2] + sin_t * (offset + lineWidth) + heightOffset

                # add translation & rotation
                axis = self.rotation[:3]
                angle = self.rotation[3]

                # Convert axis-angle → rotation object
                rot = R.from_rotvec(np.array(axis) * angle)

                # Apply rotation
                p1_rot = rot.apply([x1, y1, z1])
                p2_rot = rot.apply([x2, y2, z2])

                # Then translate
                p1_world = p1_rot + np.array(self.translation)
                p2_world = p2_rot + np.array(self.translation)

                lane_coords.append((p1_world, p2_world))

            lanes.append(lane_coords)

        return lanes

class CrossroadPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 1, 0, 0]
        self.name = "crossroad"
        self.id = ""
        self.speedLimit = -1.0
        self.shape = [[0, 0, 0], [0, 0, 1], [1, 0, 0]] 
        self.connectedRoadIDs = []
        self.boundingObject = False
        self.bottom = False
        self.appearance = "Asphalt"
        self.locked = True
        self.castShadows = False
        self.contactMaterial = "default"    
    
        # --- override with dict ---
        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__
    
class ForestPROTO:
    def __init__(self, data=None):
        self.translation = [0, 0, 0]
        self.rotation = [0, 0, 1, 0]
        self.name = "forest"
        self.id = ""
        self.shape = [[-20, -10], [20, -10], [0, 25]]
        self.density = 0.2
        self.type = "random"   
        self.maxHeight = 6
        self.minHeight = 2
        self.maxRadius = 3
        self.minRadius = 1
            
        # --- override with dict ---
        if data:
            self.update_from_dict(data)

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__

############################################################################################################################   
############################################################################################################################   
############################################################################################################################   
    
import os
import re
import matplotlib.pyplot as plt
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

def parse_waypoints_block(block):
    """
    Parse a wayPoints [ ... ] block into a list of [x, y, z] vectors.
 
    Each waypoint in a .wbt file is written as a bare triplet on its own line
    inside the brackets, e.g.:
        wayPoints [
          0 0 0
          10 0 0
          10 5 0
        ]
    """
    points = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3:
            try:
                points.append([float(p) for p in parts])
            except ValueError:
                pass
    return points

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

def parse_block(block):
    result = {}

    # -------------------------------------------------
    # lines [ ... ]
    # -------------------------------------------------
    lines_match = re.search(r"lines\s*\[(.*?)\]", block, re.DOTALL)

    if lines_match:
        result["lines"] = parse_lines_block(lines_match.group(1))

        block = (
            block[:lines_match.start()]
            + block[lines_match.end():]
        )

    # -------------------------------------------------
    # wayPoints [ ... ]
    # -------------------------------------------------
    waypoints_match = re.search(
        r"wayPoints\s*\[(.*?)\]",
        block,
        re.DOTALL
    )

    if waypoints_match:
        result["wayPoints"] = parse_vector_block(
            waypoints_match.group(1)
        )

        block = (
            block[:waypoints_match.start()]
            + block[waypoints_match.end():]
        )

    # -------------------------------------------------
    # shape [ ... ]
    # -------------------------------------------------
    shape_match = re.search(
        r"shape\s*\[(.*?)\]",
        block,
        re.DOTALL
    )

    if shape_match:
        result["shape"] = parse_vector_block(
            shape_match.group(1)
        )

        block = (
            block[:shape_match.start()]
            + block[shape_match.end():]
        )

    # -------------------------------------------------
    # scalar list blocks
    # -------------------------------------------------
    scalar_list_keys = [
        "roadBorderWidth",
        "roadTilt"
    ]
    
    connected_match = re.search(
        r"connectedRoadIDs\s*\[(.*?)\]",
        block,
        re.DOTALL
    )

    if connected_match:
        result["connectedRoadIDs"] = parse_string_list(
            connected_match.group(1)
        )

        block = (
            block[:connected_match.start()]
            + block[connected_match.end():]
        )

    for key in scalar_list_keys:

        match = re.search(
            rf"{key}\s*\[(.*?)\]",
            block,
            re.DOTALL
        )

        if match:
            result[key] = parse_scalar_list(
                match.group(1)
            )

            block = (
                block[:match.start()]
                + block[match.end():]
            )

    # -------------------------------------------------
    # remaining single-line fields
    # -------------------------------------------------
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
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    result = []

    for node_name in ["Road", "Crossroad", "Forest"]:
        blocks = extract_blocks(text, node_name)

        for b in blocks:
            parsed = parse_block(b)
            parsed["type"] = node_name  # tag the type
            result.append(parsed)

    return result
    
def extract_lanes(wbt_path):    
    parsed = extract_all_road_segments(wbt_path)
    
    roads = []
    crossroads = []
    forests = []
    
    for data in parsed: 
        if data["type"] == 'Road':
            road = RoadPROTO(data) 
            roads.append(road)
        elif data["type"] == 'Crossroad':
            crossroad = CrossroadPROTO(data)
            crossroads.append(crossroad)
        elif data["type"] == 'Forest':
            forest = ForestPROTO(data)
            forests.append(forest)
        else:
            continue

    return roads, crossroads, forests