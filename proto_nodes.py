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
                
class StraightRoadSegmentPROTO:
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
        self.startingRoadBorderWidth = 0.8
        self.endingRoadBorderWidth = 0.8

        self.rightBorder = True
        self.leftBorder = True
        self.rightBarrier = False
        self.leftBarrier = False
        self.bottom = False

        self.length = 10.0

        self.startLine = []
        self.endLine = []

        self.startingRoadTilt = 0.0
        self.endingRoadTilt = 0.0

        self.appearance = "Asphalt"
        self.pavementAppearance = "Pavement"

        self.bottomTexture = []

        self.locked = True

        self.roadBoundingObject = False
        self.rightBorderBoundingObject = False
        self.leftBorderBoundingObject = False
        self.rightBarrierBoundingObject = True
        self.leftBarrierBoundingObject = True

        self.castShadows = False
        self.contactMaterial = "default"

        # --- override with dict ---
        if data:
            self.update_from_dict(data)
            
        self.road = RoadPROTO({
            "translation": self.translation,
            "rotation": self.rotation,
            "name": self.name,
            "width": self.width,
            "numberOfLanes": self.numberOfLanes,
            "lines": self.lines,
            "roadBorderHeight": self.roadBorderHeight,
            "roadBorderWidth": [self.startingRoadBorderWidth, self.endingRoadBorderWidth],
            "rightBorder": self.rightBorder,
            "leftBorder": self.leftBorder,
            "rightBarrier": self.rightBarrier,
            "leftBarrier": self.leftBarrier,
            "bottom": self.bottom,
            "wayPoints": [[0, 0, 0], [self.length, 0, 0]],
            "roadTilt": [self.startingRoadTilt, self.endingRoadTilt],
            "startLine": self.startLine,
            "endLine": self.endLine,
            "splineSubdivision": -1,
            "appearance": self.appearance,
            "pavementAppearance": self.pavementAppearance,
            "locked": self.locked,
            "roadBoundingObject": self.roadBoundingObject,
            "rightBorderBoundingObject": self.rightBorderBoundingObject,
            "leftBorderBoundingObject": self.leftBorderBoundingObject,
            "rightBarrierBoundingObject": self.rightBarrierBoundingObject,
            "leftBarrierBoundingObject": self.leftBarrierBoundingObject,
            "contactMaterial": self.contactMaterial,
            "castShadows": self.castShadows
        })

    def update_from_dict(self, data: dict):
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return self.__dict__